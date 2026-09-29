"""Tests for the documentation website: it must never invent, leak or drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_gnn_rul_xai.docsite import catalog, figures, markup, playground, results
from thesis_gnn_rul_xai.utils.paths import project_root


@pytest.fixture(scope="module")
def root() -> Path:
    return project_root()


@pytest.fixture(scope="module")
def site_catalog(root: Path) -> catalog.Catalog:
    return catalog.build_catalog(root)


@pytest.fixture()
def stored_runs(root: Path) -> list[dict]:
    """Every training run in reports/, as the playground lists them, or skip if there are none.

    `reports/*` is deliberately not committed -- a run directory is written per
    training run and they accumulate -- so a fresh clone has none, and neither
    does the CI runner. Tests about stored runs therefore skip there for the
    same reason the data-dependent tests do: the artefact is absent, not wrong.
    """
    runs = playground.list_runs(root)
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


def test_a_page_heading_can_show_the_menu_name_and_keep_its_anchor():
    rendered = markup.render_markdown(
        "# windowing — cut each engine's history\n\ntext\n\n# A second title\n"
    )
    named = markup.name_first_heading(rendered.html, "Cutting windows")
    # The words change; the anchor a GitHub link was written against does not.
    assert '<h1 id="windowing--cut-each-engines-history">Cutting windows<a class="anchor"' in named
    assert '<h1 id="a-second-title">A second title<a class="anchor"' in named


def test_a_whole_source_file_does_not_name_its_language_twice():
    assert "code-language" not in markup.code_block("x = 1\n", "python", label=False)
    assert "code-language" in markup.code_block("x = 1\n", "python")
    assert "code-language" in markup.render_markdown("```python\nx = 1\n```\n").html


def test_the_code_panels_the_colours_are_chosen_against_are_the_stylesheets(root: Path):
    import re

    css = (root / "src/thesis_gnn_rul_xai/docsite/assets/app.css").read_text(encoding="utf-8")
    blocks = {
        "light": re.search(r":root \{(.*?)\}", css, re.DOTALL).group(1),
        "dark": re.search(r'html\[data-theme="dark"\] \{(.*?)\}', css, re.DOTALL).group(1),
    }
    for theme, block in blocks.items():
        for name, colour in markup.CODE_PANELS[theme].items():
            assert re.search(rf"--{name}: {colour};", block), (theme, name)


def test_every_code_colour_reads_on_its_own_panel_in_both_themes():
    """CLAUDE.md rule 11 asks for 4.5 to 1. The light scheme's colours once
    showed through on the dark panel, where blue measured 2.01 to 1, and a
    light background behind every line of dark-theme code made it unreadable."""
    import re

    sheet = markup.pygments_stylesheet()
    rules = sheet.strip().splitlines()
    assert rules
    for rule in rules:
        theme = "dark" if rule.startswith('html[data-theme="dark"]') else "light"
        prefix = 'html[data-theme="dark"] ' if theme == "dark" else ""
        selectors = rule.split(" {")[0].split(", ")
        # Only code is coloured: a document's code blocks and a notebook's code cells.
        assert [s.rsplit(" .", 1)[0] for s in selectors] == [
            f"{prefix}.code-block pre code",
            f"{prefix}.notebook-frame .highlight pre",
        ], rule
        assert "background" not in rule, rule
        for colour in re.findall(r"color: (#[0-9a-f]{6})", rule):
            ratio = markup.contrast_ratio(colour, markup.CODE_PANELS[theme]["panel"])
            assert ratio >= 4.5, (rule, round(ratio, 2))


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
    taught = [doc for doc in learn.documents if doc.kind == "learn"]
    assert [(doc.title, doc.depth) for doc in taught] == [
        ("Start here", 0), ("Data", 0), ("Preprocessing", 0),
        ("Temporal algorithm", 0), ("GNN", 0), ("Architecture and training", 1),
        ("XAI", 0), ("Dynamic SHAP", 1), ("GNN-LRP", 1),
        ("Follow one real example", 0), ("Reading a result", 0),
        ("Paper, choice or measurement", 0),
    ]
    assert all(doc.words > 200 for doc in taught)


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


def test_the_menu_has_the_playground_and_four_sections(site_catalog: catalog.Catalog):
    assert site_catalog.primary is not None
    assert site_catalog.primary.path == "explore/playground"
    assert [section.title for section in site_catalog.sections] == [
        "Learn the method",
        "Notebooks",
        "Code",
        "Choices and checks",
    ]
    # The playground is the one page drawn here rather than read from a file.
    explore = [doc.path for doc in site_catalog.by_path.values() if doc.kind == "explore"]
    assert explore == ["explore/playground"]


def test_pages_that_repeat_other_pages_are_not_in_the_menu(
    site_catalog: catalog.Catalog, root: Path
):
    listed = set(site_catalog.by_path)
    # The files stay in the repository; only the menu leaves them out.
    for path in ("README.md", "docs/AUDIT_2026-08-31_paper_against_code.md"):
        assert (root / path).is_file(), path
        assert path not in listed, path
    code = next(s for s in site_catalog.sections if s.key == "code")
    assert ("data/raw/README.md", "Getting the NASA data") in {
        (doc.path, doc.title) for doc in code.documents
    }
    checks = next(s for s in site_catalog.sections if s.key == "checks")
    assert checks.documents[0].title == "Assumptions log"


def test_every_listed_page_has_a_first_heading_to_show_its_menu_name(
    site_catalog: catalog.Catalog, root: Path
):
    for doc in site_catalog.by_path.values():
        if not doc.path.endswith(".md"):
            continue
        rendered = markup.render_markdown(
            catalog.read_document(doc.path, root), source_path=doc.path
        )
        assert any(heading.level == 1 for heading in rendered.headings), doc.path


def test_reading_a_file_outside_the_project_is_refused(root: Path):
    with pytest.raises(ValueError, match="outside the project"):
        catalog.read_document("../../../etc/passwd", root)


def test_search_finds_a_phrase_that_is_certainly_there(
    site_catalog: catalog.Catalog, root: Path
):
    hits = catalog.search("piecewise", site_catalog, root)
    assert hits
    assert all(set(hit) == {"path", "title", "context"} for hit in hits)


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


def test_an_entry_superseded_in_its_body_is_picked_up(tmp_path: Path):
    """A status can sit in an entry's body rather than its title, and must count.

    A-06 and A-18 used that form while A-22 had removed the validation split.
    Both went active again on 2026-09-02, so no entry in the log exercises the
    branch any more, and it is checked here against a log written in the test
    rather than left uncovered until some future entry needs it.
    """
    log = tmp_path / "docs" / "ASSUMPTIONS.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "# Assumptions log\n\n"
        "## A-99 — A title carrying no status word\n\n"
        "**Superseded 2026-08-29 by A-22.** The reproduction holds nothing "
        "out.\n",
        encoding="utf-8",
    )
    entries = {entry.identifier: entry for entry in results.load_entries(tmp_path)}
    assert entries["A-99"].status == "superseded"


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
    # A-06 and A-18 were here until 2026-09-02, when the validation split they
    # describe came back and both entries became current again.
    assert retired == {"A-08"}


def test_every_stored_run_names_the_commit_that_produced_it(stored_runs):
    for run in stored_runs:
        assert run["commit"] and run["commit"] != "unknown", run["name"]
        assert run["dirty"] == run["commit"].endswith("-dirty")


def test_a_single_seed_run_keeps_its_missing_spread_as_missing(stored_runs):
    for run in stored_runs:
        if run["seeds"] == 1:
            assert (run["std"] or {}).get("rmse") is None
            return
    pytest.skip("no single-seed run is stored")


def test_the_directory_name_of_every_run_carries_its_commit(stored_runs):
    for run in stored_runs:
        assert run["commit"] in run["name"], run["name"]


def test_a_verification_report_is_not_listed_as_a_training_run(stored_runs):
    # scripts/verify_claims.py writes into reports/ too, with no scores in it.
    # Listing one as a run would show a row of dashes that reads like a
    # training run that failed.
    directories = {run["name"] for run in stored_runs}
    assert not any(name.startswith("verification_") for name in directories)


def test_every_listed_run_has_the_four_measures(stored_runs):
    for run in stored_runs:
        assert set(run["mean"]) >= {"rmse", "score", "r2", "asue"}, run["name"]
        assert run["mean"]["rmse"] is not None, run["name"]


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


def test_the_correlation_figure_is_the_papers_eighteen_by_eighteen_table(raw_dir: Path):
    """The paper's Fig. 5 has eighteen rows: engine, cycle, two settings, fourteen sensors."""
    import numpy as np

    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS

    labels, matrix = figures.correlation_table()
    assert len(labels) == 18
    assert matrix.shape == (18, 18)
    assert labels[:2] == ["unit", "cycle"]
    assert labels[-14:] == list(PAPER_SENSOR_SYMBOLS)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    drawn = figures.render_figure("correlation")
    assert drawn.startswith("<svg")
    assert "figure-notice" not in drawn
    for label in labels:
        assert label in drawn


def _node_groups(drawn: str) -> dict[str, str]:
    """Each sensor drawn in a graph figure, with the markup of its own group."""
    import re

    groups = {}
    for group in re.findall(r"<g>(.*?)</g>", drawn, flags=re.S):
        # The sensor's name is the one bold label inside its group.
        name = re.search(r'font-weight="600"[^>]*>([^<]+)</text>', group)
        if name:
            groups[name.group(1)] = group
    return groups


def _ringed(drawn: str) -> set[str]:
    """The sensors a graph figure draws a dashed ring round."""
    return {
        name for name, group in _node_groups(drawn).items() if "stroke-dasharray" in group
    }


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
    width = prepared.train.features.shape[2]
    assert f"({spaced(len(prepared.train))}, 14, {width})" in drawn
    assert f"({spaced(len(prepared.validation))}, 14, {width})" in drawn
    assert f"({spaced(len(prepared.test))}, 14, {width})" in drawn
    # The total before the split, and the size of the graph the same run built.
    assert spaced(windows) in drawn
    nodes = prepared.diagnostics["graph"]["n_nodes"]
    assert f"{nodes} by {nodes}" in drawn


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

    body = notebooks.render_notebook(
        root / "notebooks" / "01_preprocessing_fd001.ipynb"
    )
    assert "What notebook 01 does" in body, (
        "the pipeline map is not in the page the site builds from notebook 01"
    )


def test_the_message_passing_figure_stops_where_the_graph_stops():
    """The sensors marked as never reached are exactly those more than three edges from P15."""
    from thesis_gnn_rul_xai.graph.adjacency import (
        adjacency_report,
        build_physics_adjacency,
        pairs_beyond_reach,
    )

    matrix, symbols = build_physics_adjacency()
    far = {
        b if a == "P15" else a
        for a, b, _ in pairs_beyond_reach(matrix, symbols, 3)
        if "P15" in (a, b)
    }
    assert far, "the graph no longer has a sensor beyond three layers from P15"
    drawn = figures.render_figure("message-passing")
    groups = _node_groups(drawn)
    assert len(groups) == len(symbols)
    assert {name for name, group in groups.items() if "never" in group} == far
    # The sensor whose removal splits the graph is ringed here as well, so a
    # page can show this drawing without the plain one beside it.
    report = adjacency_report(matrix, symbols)
    assert _ringed(drawn) == set(report["articulation_points"])


def test_the_architecture_figure_prints_the_real_parameter_counts():
    """Each box's count is read from the built network, with the stage and Table 4's Input as one box."""
    import re

    pytest.importorskip("torch")
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.graph.adjacency import (
        build_physics_adjacency,
        normalize_adjacency,
    )
    from thesis_gnn_rul_xai.models.estgnn import ESTGNN

    matrix, _ = build_physics_adjacency()
    model = ESTGNN(
        normalize_adjacency(matrix, method="row"),
        window_size=PreprocessingConfig().window_size,
    )
    rows = model.layer_summary()
    assert [rows[0]["layer"], rows[1]["layer"]] == ["Temporal", "Input"]
    expected = [("Input", rows[0]["parameters"])] + [
        (row["layer"], row["parameters"]) for row in rows[2:]
    ]

    drawn = figures.render_figure("architecture")
    assert "figure-notice" not in drawn
    # Every box starts with its rectangle; its name is the 13-unit label and its
    # count the bold 10-unit one.
    boxes = drawn.split("<rect ")[1:]
    drawn_counts = [
        (
            re.search(r'font-size="13"[^>]*>([^<]+)</text>', box).group(1),
            re.search(r'font-size="10" font-weight="600"[^>]*>([^<]+)</text>', box).group(1),
        )
        for box in boxes
    ]
    assert drawn_counts == [
        (layer, f"{int(count):,}".replace(",", " ")) for layer, count in expected
    ]
    assert "w/o LSTM" in boxes[0]


# -- the taught pages ------------------------------------------------------


def test_every_figure_a_taught_page_asks_for_exists(root: Path):
    import re

    known = set(figures.figure_names())
    for page in sorted((root / "docs" / "learn").glob("*.md")):
        for name in re.findall(r"\{\{\s*figure:([a-z0-9-]+)\s*\}\}", page.read_text(encoding="utf-8")):
            assert name in known, f"{page.name} asks for unknown figure {name!r}"


def test_sensor_selection_uses_real_column_names_without_invented_readings():
    import xml.etree.ElementTree as ET

    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS, SENSORS

    svg = ET.fromstring(figures.draw_sensor_selection())
    groups = list(svg.iter("{http://www.w3.org/2000/svg}g"))
    source = [g for g in groups if "select-source" in g.attrib.get("class", "")]
    picked = [g for g in groups if g.attrib.get("class") == "select-picked"]
    ns = "{http://www.w3.org/2000/svg}text"
    assert [g.find(ns).text for g in source] == [s.symbol for s in SENSORS]
    assert [g.find(ns).text for g in picked] == list(PAPER_SENSOR_SYMBOLS)
    assert all(len(list(g.iter(ns))) == 1 for g in source + picked)
    source_sizes = {
        g.find(ns).text: g.find("{http://www.w3.org/2000/svg}rect").attrib for g in source
    }
    for column in picked:
        original = source_sizes[column.find(ns).text]
        selected = column.find("{http://www.w3.org/2000/svg}rect").attrib
        assert (selected["width"], selected["height"]) == (original["width"], original["height"])
        assert 250 + float(selected["height"]) < float(svg.attrib["viewBox"].split()[3])


def test_old_xai_section_links_redirect_to_existing_subpage_headings(root: Path):
    import re

    script = (root / "src/thesis_gnn_rul_xai/docsite/assets/app.js").read_text(encoding="utf-8")
    mapping = script.split("var XAI_MOVED_SECTIONS = {", 1)[1].split("};", 1)[0]
    for anchor, name in re.findall(r'"([^"]+)": "([^"]+)"', mapping):
        page = root / "docs/learn" / name
        rendered = markup.render_markdown(page.read_text(encoding="utf-8"), source_path=page.as_posix())
        assert anchor in {heading.anchor for heading in rendered.headings}


def test_a_figure_placeholder_inside_code_stays_text(root: Path):
    """The page turns {{figure:name}} into a figure only where it stands as a
    paragraph of its own; one written inside code, to show the syntax, once
    became an error box in the middle of a sentence."""
    import re

    script = (root / "src/thesis_gnn_rul_xai/docsite/assets/app.js").read_text(encoding="utf-8")
    pattern = re.compile(re.search(r"var FIGURE_PATTERN = /(.+)/gi;", script).group(1), re.IGNORECASE)
    html = markup.render_markdown(
        "{{figure:smoothing}}\n\nWrite `{{figure:name}}` on a line of its own.\n", source_path="x.md").html
    assert pattern.findall(html) == ["smoothing"]


def test_the_menu_becomes_a_drawer_at_one_width_everywhere(root: Path):
    """The stylesheet, the first-paint script and the page's script each decide
    where the menu turns into a drawer; they must agree, or the menu opens
    beside the page in one and over it in another."""
    import re

    assets = root / "src/thesis_gnn_rul_xai/docsite/assets"
    widths = [
        re.search(r'var narrowScreen = window\.matchMedia\("\(max-width: (\d+)px\)"\)',
                  (assets / "app.js").read_text(encoding="utf-8")).group(1),
        re.search(r'var narrow = window\.matchMedia\("\(max-width: (\d+)px\)"\)',
                  (assets / "shell.html").read_text(encoding="utf-8")).group(1),
        re.search(r"@media \(max-width: (\d+)px\) \{\s*\.sidebar \{\s*position: fixed",
                  (assets / "app.css").read_text(encoding="utf-8")).group(1),
    ]
    assert widths == ["1199"] * 3
    # The open menu takes the contents list's room only where it stands
    # beside the page; as a drawer over the page it takes none.
    beside = re.search(r'@media \(min-width: (\d+)px\) and \(max-width: \d+px\) \{ html\[data-nav="open"\] \.toc',
                       (assets / "app.css").read_text(encoding="utf-8")).group(1)
    assert int(beside) == int(widths[0]) + 1


def test_desktop_scroll_is_contained_and_resets_when_a_document_changes(root: Path):
    import re

    assets = root / "src/thesis_gnn_rul_xai/docsite/assets"
    css = (assets / "app.css").read_text(encoding="utf-8")
    script = (assets / "app.js").read_text(encoding="utf-8")
    frame = css.split("/* Desktop uses one screen-sized frame.", 1)[1]
    desktop = re.search(r"@media \(min-width: (\d+)px\)", frame).group(1)
    assert f'var desktopFrame = window.matchMedia("(min-width: {desktop}px)")' in script
    assert "body { height: 100dvh; display: flex; flex-direction: column; overflow: hidden; }" in frame
    assert ".layout { flex: 1 1 0; min-height: 0; align-items: stretch; overflow: hidden; }" in frame
    assert ".content { min-height: 0; overflow-y: auto;" in frame
    assert ".content:has(.pg-design) { height: auto; min-height: 0; overflow: hidden; }" in frame
    assert 'content.scrollTo({ top: 0, behavior: "instant" })' in script
    assert "root: desktopFrame.matches ? content : null" in script
    assert ".pg-data-table-wrap { position: relative;" in css
    assert ".pg-data-table caption.visually-hidden { top: 0; left: 0; }" in css


def test_the_page_scripts_parse_in_older_safari(root: Path):
    """A lookbehind in a regular expression stops Safari before 16.4 from
    reading the whole file it is in, and with it every page of the site."""
    assets = root / "src/thesis_gnn_rul_xai/docsite/assets"
    for name in ("app.js", "playground.js"):
        text = (assets / name).read_text(encoding="utf-8")
        assert "(?<=" not in text and "(?<!" not in text, name


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
    assert set(state) >= {"commit", "torch", "jupyter", "hasFD001", "stale"}
    assert isinstance(state["hasFD001"], bool)
    assert isinstance(state["stale"], bool)
    assert json.dumps(state)  # the browser has to be able to read it


@pytest.fixture(scope="module")
def live_site(root: Path):
    """The site serving this repository on a free port, stopped afterwards."""
    import threading
    from http.server import ThreadingHTTPServer

    from thesis_gnn_rul_xai.docsite.server import DocsHandler

    handler = type("RepositoryHandler", (DocsHandler,), {"root": root})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, dict]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read())


def test_a_menu_page_goes_by_its_menu_name_in_its_heading_and_its_tab(live_site: str):
    status, doc = _get(f"{live_site}/api/doc?path=docs/learn/01_the_problem.md")
    assert status == 200
    assert doc["title"] == "Data"
    first = doc["html"].split("</h1>")[0]
    assert first.startswith("<h1 ") and ">Data<a" in first
    assert "words" not in doc  # the reading time is kept, its word count is not


def test_a_page_outside_the_menu_keeps_its_own_heading(live_site: str, root: Path):
    status, doc = _get(f"{live_site}/api/doc?path=README.md")
    assert status == 200
    own = markup.render_markdown(catalog.read_document("README.md", root)).title
    assert doc["title"] == own
    assert doc["minutes"] is None


def test_a_notebook_page_goes_by_its_menu_name(live_site: str):
    pytest.importorskip("nbconvert")
    status, notebook = _get(
        f"{live_site}/api/notebook?path=notebooks/xai/02_occlusion_fd001.ipynb"
    )
    assert status == 200
    assert notebook["title"] == "Occlusion: hiding one sensor"
    assert "css" not in notebook  # its code is coloured by the site's own stylesheet
    assert notebook["html"].count("<h1") == 1
    assert ">Occlusion: hiding one sensor<a class=\"anchor-link\"" in notebook["html"]


def test_a_source_file_is_named_by_its_path_and_shown_without_a_language_badge(
    live_site: str,
):
    path = "src/thesis_gnn_rul_xai/graph/adjacency.py"
    status, file = _get(f"{live_site}/api/source?path={path}")
    assert status == 200
    assert file["title"] == path
    assert "code-language" not in file["html"]
    assert '<span class="k">' in file["html"]  # still coloured


def test_the_routes_only_the_removed_pages_used_are_gone(live_site: str):
    for route in ("/api/graph", "/api/model"):
        status, answer = _get(live_site + route)
        assert status == 404, route
        assert "No route" in answer["error"]
    status, log = _get(f"{live_site}/api/results")
    assert status == 200
    # What the entry links and the log's re-check result need, and nothing more.
    assert set(log) == {"entries", "verification"}
    assert log["entries"]


def test_every_old_address_is_sent_to_a_page_that_exists(root: Path):
    import re

    script = (root / "src/thesis_gnn_rul_xai/docsite/assets/app.js").read_text(encoding="utf-8")
    moved = dict(re.findall(r'"((?:explore/)?[a-z]+)": "(#/[^"]+)"', script))
    assert set(moved) == {
        "explore/graph",
        "figures",
        "explore/architecture",
        "explore/results",
        "explore/assumptions",
    }
    listed = catalog.build_catalog(root).by_path
    for old, new in moved.items():
        target = new.removeprefix("#/doc/").removeprefix("#/")
        assert target in listed or (root / target).is_file(), (old, new)


def test_a_second_server_cannot_take_a_port_another_server_already_holds():
    """Windows would otherwise let both listen, and the older one can answer with older code."""
    from thesis_gnn_rul_xai.docsite.server import DocsHandler, DocsServer

    first = DocsServer(("127.0.0.1", 0), DocsHandler)
    try:
        with pytest.raises(OSError):
            # If this ever succeeds the port is shared, which is the bug; close
            # it so the test fails on the assertion rather than on a leak.
            DocsServer(("127.0.0.1", first.server_address[1]), DocsHandler).server_close()
    finally:
        first.server_close()


def test_the_site_notices_when_its_own_code_changed_after_it_started():
    """A server loads its Python once; a file edited after that is a change it is not serving."""
    import time

    from thesis_gnn_rul_xai.docsite.server import code_changed_since

    assert code_changed_since(0.0) is True
    assert code_changed_since(time.time() + 3600) is False
