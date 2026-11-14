"""Tests for the playground: it may only run what the scripts run, and must say where every number came from."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from thesis_gnn_rul_xai.docsite import catalog, playground, server
from thesis_gnn_rul_xai.evaluation import published
from thesis_gnn_rul_xai.utils.paths import project_root

ROOT = project_root()


# -- the paper's figures -----------------------------------------------------


def test_the_transcribed_rmse_row_averages_to_what_table_8_prints():
    """A typo in one transcribed figure would move the average the paper prints."""
    full = [published.published_row("full", s).rmse for s in published.SUBSETS]
    assert round(sum(full) / 4, 2) == published.ABLATION_ROWS["full_average"]["rmse"]


@pytest.mark.parametrize("key", ["without_gcn", "without_lstm", "without_physics_prior"])
def test_each_ablation_row_matches_the_average_and_delta_table_11_prints(key):
    rows = [published.published_row(key, s).rmse for s in published.SUBSETS]
    printed = published.ABLATION_ROWS[key]
    assert round(sum(rows) / 4, 2) == printed["average"]
    delta = published.published_row(key, "FD001").rmse - published.published_row("full", "FD001").rmse
    assert round(delta, 2) == printed["FD001_delta"]


def test_table_11_prints_rmse_only_so_nothing_else_is_invented_for_it():
    row = published.published_row("without_lstm", "FD002")
    assert (row.rmse, row.score, row.r2, row.asue) == (23.34, None, None, None)


def test_table_12_covers_the_paper_sensors_once_each_at_fifteen_graphs():
    from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS

    assert set(published.SHAP_TABLE_12) == set(PAPER_SENSOR_SYMBOLS)
    assert sorted(published.FIG_10_RING) == sorted(PAPER_SENSOR_SYMBOLS)
    assert len(published.SHAP_TABLE_12_GRAPHS) == 15
    assert all(len(v) == 15 for v in published.SHAP_TABLE_12.values())


def test_table_12_column_sums_are_the_ones_finding_f08_prints():
    """F-08 summed seven of the printed columns; a typo in one cell would move its sum."""
    printed = {1: 45.51, 2: 41.33, 5: 33.92, 82: 26.38, 189: 18.16, 190: 16.67, 191: 17.60}
    for graph, total in printed.items():
        column = published.SHAP_TABLE_12_GRAPHS.index(graph)
        assert round(sum(v[column] for v in published.SHAP_TABLE_12.values()), 2) == total


def test_table_12_top_four_in_each_column_are_the_cells_the_paper_sets_in_bold():
    """The bold cells were read from the font of each cell in the PDF on 2026-09-14."""
    bold = {
        **{g: {"T50", "Nc", "Ps30", "phi"} for g in (1, 2, 3, 4, 5, 191)},
        **{g: {"T50", "P30", "Ps30", "phi"} for g in (78, 79, 80, 81, 82)},
        **{g: {"T50", "Nc", "Ps30", "BPR"} for g in (187, 188, 189, 190)},
    }
    for column, graph in enumerate(published.SHAP_TABLE_12_GRAPHS):
        ranked = sorted(published.SHAP_TABLE_12, key=lambda s: -published.SHAP_TABLE_12[s][column])
        assert set(ranked[:4]) == bold[graph], graph


# -- settings ------------------------------------------------------------------


def test_every_default_passes_its_own_check():
    settings = playground.validate_settings({})
    assert settings == {f["key"]: f["default"] for f in playground.FIELDS}


def test_every_number_default_lies_on_its_fields_steps():
    """A browser's number field allows only min + k * step, for a whole k, and
    marks anything else invalid; its arrow keys then can never land on the
    default again. Three fields once held their defaults off that grid."""
    import math

    for spec in playground.FIELDS:
        if spec["kind"] not in ("int", "float", "widths"):
            continue
        step = spec.get("step", 1)
        defaults = spec["default"] if spec["kind"] == "widths" else [spec["default"]]
        for value in defaults:
            k = (value - spec["min"]) / step
            assert math.isclose(k, round(k), abs_tol=1e-9), (spec["key"], value, spec["min"], step)


def test_every_setting_says_what_it_is_for_in_plain_words():
    """An i says first what its setting is for, then, under a line, where the
    default comes from. The plain part never cites the paper or the log, since
    the part under the line does that, and it stays short enough to read in a
    card. An option with an i of its own says what that option does."""
    import re

    cited = re.compile(r"\b[ADFP]-\d+\b|\bTable \d|\bEq\. ?\d|\bSection \d|\bFig\. ?\d")
    texts = [(f["key"], f["about"]) for f in playground.FIELDS]
    texts += [(f"{f['key']}={c['value']}", c["about"]) for f in playground.FIELDS
              for c in f.get("choices", []) if c["source"] is not None]
    texts += [(f"procedure={p['key']}", p["about"]) for p in playground.PROCEDURES]
    texts += [("procedure", playground.PROCEDURE_ABOUT)]
    for name, about in texts:
        assert about and about == about.strip(), name
        assert not cited.search(about), (name, cited.search(about).group(0))
        assert len(about.split()) <= 100, (name, len(about.split()))
    assert playground.options()["procedure_about"] == playground.PROCEDURE_ABOUT


def test_every_read_more_leads_to_a_heading_that_exists():
    """The Read more at the foot of an i opens a part of the documentation in a
    new tab; a heading renamed in a document would leave it pointing nowhere."""
    import re

    from thesis_gnn_rul_xai.docsite import results
    from thesis_gnn_rul_xai.docsite.markup import slugify

    known = {entry.identifier for entry in results.load_entries(ROOT)}
    for key in playground.settings._READ_MORE:
        address = playground.read_more(key)
        if address.startswith("#/entry/"):
            assert address.removeprefix("#/entry/") in known, key
            continue
        path, anchor = address.removeprefix("#/doc/").split("#")
        text = (ROOT / path).read_text(encoding="utf-8")
        seen: dict[str, int] = {}
        slugs = {slugify(m.group(1), seen) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)}
        assert anchor in slugs, (key, path, anchor)
    # Every setting and every option with an i of its own has one.
    for field in playground.FIELDS:
        assert field["key"] in playground.settings._READ_MORE, field["key"]


def test_where_a_default_comes_from_is_said_once_per_setting():
    """A source sits on the row or on its options, never both, so the page draws one tag per fact."""
    kinds = {"paper", "assumption", "project", "extension"}
    untagged_rows = set()
    for field in playground.FIELDS:
        sources = [field["source"]] + [c["source"] for c in field.get("choices", [])]
        for source in sources:
            if source is None:
                continue
            assert source["kind"] in kinds, field["key"]
            if source["kind"] in ("paper", "assumption"):
                assert source["ref"], field["key"]
        if field["source"] is None:
            untagged_rows.add(field["key"])
        else:
            assert all(c["source"] is None for c in field.get("choices", [])), field["key"]
    # Three settings whose options come from different places, and two nothing sets.
    assert untagged_rows == {"graph", "temporal", "spatial", "seed", "device"}
    for key in ("graph", "temporal", "spatial"):
        spec = next(f for f in playground.FIELDS if f["key"] == key)
        assert all(c["source"] is not None for c in spec["choices"]), key


def test_the_procedure_is_the_only_control_for_what_is_held_back():
    groups = {f["key"]: f["group"] for f in playground.FIELDS}
    assert {key for key, group in groups.items() if group == "procedure"} == {
        "validation_fraction", "early_stopping", "early_stopping_patience"}
    # Not a tab of the form: the page draws these inside Training, under
    # the choice of procedure.
    assert "procedure" not in {key for key, _ in playground.GROUPS}
    assert "training" in {key for key, _ in playground.GROUPS}
    assert dict(playground.GROUPS)["training"] == "Training config"
    paper, held_back = playground.PROCEDURES
    assert paper["settings"] == {"validation_fraction": 0.0, "early_stopping": False}
    assert held_back["label"] == "Held back, best pass"
    assert held_back["settings"] == {"validation_fraction": 0.2, "early_stopping": True}
    for procedure in playground.PROCEDURES:
        playground.validate_settings(procedure["settings"])


def test_the_pipeline_takes_out_one_part_at_a_time_as_table_11_does():
    """The page's pipeline takes a part out by setting the value its cross
    names. Those values must be the settings of the variants of Table 11, each
    of which differs from the full model in one setting, or a stored run would
    be set beside a row the pipeline did not build."""
    import re

    script = (ROOT / "src/thesis_gnn_rul_xai/docsite/assets/playground.js").read_text(encoding="utf-8")
    removed = dict(re.findall(r'(\w+): "(\w+)"', re.search(r"var REMOVED = \{(.*?)\};", script).group(1)))
    variants = {a["key"]: a["settings"] for a in playground.ABLATIONS}
    full = variants["full"]
    defaults = {f["key"]: f["default"] for f in playground.FIELDS}
    # The pipeline opens on the full model.
    assert all(defaults[key] == value for key, value in full.items())
    taken_out = {}
    for key in ("without_lstm", "without_physics_prior", "without_gcn"):
        moved = {k: v for k, v in variants[key].items() if full[k] != v}
        assert len(moved) == 1, key
        taken_out.update(moved)
    assert removed == taken_out == {"temporal": "last_cycle", "graph": "complete", "spatial": "none"}
    # A network without graph layers uses no sensor graph, so its row names none.
    assert "graph" not in variants["without_gcn"]
    # The sensor graph is a step of the graph layers' own flowchart, so its
    # setting is in their group, where that step's link leads.
    groups = {f["key"]: f["group"] for f in playground.FIELDS}
    assert groups["graph"] == groups["adjacency_normalization"] == groups["gcn_hidden"] == "graph"


@pytest.mark.parametrize("graph", ["physics", "complete", "correlation"])
def test_the_sensor_graph_is_taken_out_from_one_step_of_the_graph_box(graph):
    """The page puts the cross that takes the sensor graph out, and the + that
    puts it back, on the graph box's step whose value links to the graph
    setting. Exactly one step must, whichever graph is chosen, or the control
    would vanish or appear twice."""
    pytest.importorskip("torch")
    answer = _flowchart("graph", graph=graph)
    holding = [step["name"] for step in _steps(answer)
               if any(part.get("setting") == "graph" for part in step.get("parts") or [])]
    assert holding == ["Sensor graph"]


# -- a box of the pipeline, as its own flowchart ----------------------------------

TEMPORAL_CHOICES = ["lstm", "gru", "last_cycle", "mean_window", "window_summary", "raw_window"]


def _flowchart(stage: str, **settings) -> dict:
    return playground.stage_detail(stage, playground.validate_settings(settings))


def _steps(answer: dict) -> list[dict]:
    """Every step of a flowchart, the ones drawn side by side included, without what enters or leaves."""
    flat = []
    for step in answer["steps"]:
        flat += step["steps"] if step.get("kind") in ("group", "branch", "evaluation") else [step]
    return [step for step in flat if step.get("kind") != "edge"]


def _step(answer: dict, name: str) -> dict:
    return next(step for step in _steps(answer) if step["name"] == name)


@pytest.mark.parametrize("stage,settings", [
    ("data", {}),
    ("preprocessing", {}),
    ("preprocessing", {"validation_fraction": 0.0, "early_stopping": False}),
    *(("temporal", {"temporal": kind}) for kind in TEMPORAL_CHOICES),
    *(("temporal", {"temporal": kind, "shared_across_sensors": True,
                     "temporal_out_features": 2}) for kind in ("lstm", "gru")),
    ("graph", {}),
    ("graph", {"graph": "complete", "adjacency_normalization": "symmetric",
               "gcn_hidden": [1, 2, 3], "fc_hidden": 1, "gcn_bias": True}),
    ("graph", {"graph": "correlation"}),
])
def test_diagram_links_open_matching_tutorial_sections(stage, settings, raw_dir):
    """Every displayed operation lands on its own named input-to-output example."""
    import re

    from thesis_gnn_rul_xai.docsite.markup import render_markdown

    pytest.importorskip("torch")
    answer = _flowchart(stage, **settings)
    assert "error" not in answer
    documents = {}
    for step in _steps(answer):
        path, anchor = step["more"].removeprefix("#/doc/").split("#")
        if path not in documents:
            text = (ROOT / path).read_text(encoding="utf-8")
            documents[path] = render_markdown(text, source_path=path)
        rendered = documents[path]
        heading = next(h for h in rendered.headings if h.anchor == anchor)
        assert heading.text == playground.settings._STEP_HEADINGS.get(step["name"], step["name"])
        section = rendered.html.split(f'id="{anchor}"', 1)[1]
        section = re.split(rf"<h[1-{heading.level}]\b", section, maxsplit=1)[0]
        for label in ("Input", "Function", "Output", "Example"):
            assert re.search(rf"\b{label}\b", section, re.I), (step["name"], label)


@pytest.mark.parametrize("changes", [
    {},
    {"normalization": "minmax_global", "temporal": "gru", "shared_across_sensors": True},
    {"normalization": "minmax_per_condition", "temporal": "last_cycle", "smoothing": False,
     "validation_fraction": 0.0, "early_stopping": False},
    {"temporal": "mean_window", "graph": "complete", "adjacency_normalization": "symmetric"},
    {"temporal": "window_summary", "graph": "correlation"},
    {"temporal": "raw_window", "gcn_hidden": [1, 2, 3], "gcn_bias": True},
    {"temporal": "lstm", "shared_across_sensors": True},
    {"temporal": "gru", "shared_across_sensors": False},
    {"spatial": "none"},
    {"spatial": "none", "temporal": "raw_window"},
])
def test_diagram_cards_share_matching_help_and_explain_every_output(raw_dir, changes):
    """All diagram layouts explain their outputs and keep matching operation help and tutorial links."""
    import shutil

    pytest.importorskip("torch")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the diagram's rendered links.")
    settings = playground.validate_settings(changes)
    flows = {stage: playground.stage_detail(stage, settings) for stage in playground.STAGES}
    payload = {"settings": settings, "options": playground.options(), "flows": flows,
               "diagrams": {stage: _steps(answer) for stage, answer in flows.items()}}
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {window: {}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {S, SECTIONS, sectionHtml, vstepHtml, vflowHtml, sectionForStep, sectionName, sectionHelp, optionNote,
    setup: function (helpers) { T = helpers; }};`), context);
const P = context.window.test;
const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
P.setup({escapeHtml, fixed: (x, n) => Number(x).toFixed(n)});
P.S.settings = payload.settings; P.S.infos = 0; P.S.restore = {}; P.S.options = payload.options;
P.S.fields = Object.fromEntries(payload.options.fields.map(f => [f.key, f]));
for (const [group, sections] of Object.entries(P.SECTIONS)) {
  sections.forEach((section, index) => {
    const html = P.sectionHtml(section, index, group);
    assert.equal((html.match(/class="pg-info-button"/g) || []).length, 1,
      group + ': only the section title should have an info button');
    assert.ok(!html.slice(html.indexOf('<div class="pg-fields">')).includes('class="pg-info-button"'),
      group + ': an individual setting or option still has an info button');
  });
}
function firstCard(html) {
  const start = html.indexOf('class="pg-info-card">') + 'class="pg-info-card">'.length;
  const content = html.slice(start);
  let depth = 1;
  for (const tag of content.matchAll(/<\/?span\b[^>]*>/g)) {
    depth += tag[0].startsWith('</') ? -1 : 1;
    if (depth === 0) return content.slice(0, tag.index);
  }
  throw new Error('Unclosed information card');
}
for (const [stage, steps] of Object.entries(payload.diagrams)) {
  P.S.zoom = stage;
  const flow = P.vflowHtml(payload.flows[stage].steps);
  const shapes = [...flow.matchAll(/<code class="pg-vshape">([^<]+)<\/code>/g)];
  assert.ok(shapes.length, stage + ' must display its output dimensions');
  for (const shape of shapes) {
    const afterShape = flow.slice(shape.index + shape[0].length);
    assert.match(afterShape, /^<span class="pg-info[^\"]*"><button [^>]*aria-label="About outputs after /,
      stage + ': missing output info beside ' + shape[1]);
    const card = firstCard(afterShape);
    assert.ok(card.includes('class="pg-transfer-output"'), stage + ': missing output');
    assert.match(card, /class="pg-transfer-axes">[^<]+</, stage + ': missing dimension meanings');
    if (shape[1].includes('B')) {
      assert.ok(card.includes('B is the number of windows'), stage + ': unexplained B');
    }
  }
  for (const step of steps) {
    const html = P.vstepHtml(step, false);
    const card = firstCard(html);
    assert.ok(card.includes('href="' + escapeHtml(step.more) + '"'), step.name);
    const section = P.sectionForStep(step);
    if (section && section.fields.length === 1 && section.fields[0] === 'temporal') {
      const choice = P.S.fields.temporal.choices.find(c => c.value === payload.settings.temporal);
      assert.equal(card, P.optionNote(choice).replace(/<a /g, '<a tabindex="-1" '), step.name);
    } else if (section) {
      assert.equal(card, P.sectionHelp(section, step.more).replace(/<a /g, '<a tabindex="-1" '), step.name);
    } else {
      assert.ok(card.includes(escapeHtml(step.does)), step.name);
    }
    assert.ok(card.includes('target="_blank" rel="noopener"'), step.name);
  }
}
"""
    rendered = subprocess.run([node, "-e", script], cwd=ROOT, input=json.dumps(payload),
                              text=True, encoding="utf-8", capture_output=True)
    assert rendered.returncode == 0, rendered.stderr


def test_design_controls_and_information_cards_keep_the_requested_layout():
    """The visible layer table, card footer and stateful reset are one UI contract."""
    import re

    assets = ROOT / "src/thesis_gnn_rul_xai/docsite/assets"
    script = (assets / "playground.js").read_text(encoding="utf-8")
    css = (assets / "app.css").read_text(encoding="utf-8")

    assert '<section class="pg-layers"' in script
    assert '<h4 id="pg-layers-title">Layer by layer</h4>' in script
    assert 'details class="pg-details" data-layers' not in script
    assert 'class="pg-info-footer"' in script
    footer = re.search(r"\.pg-info-footer\s*\{([^}]+)\}", css).group(1)
    settings_foot = re.search(r"\.pg-settings-foot\s*\{([^}]+)\}", css).group(1)
    assert "display: flex" in footer and "justify-content: space-between" in footer
    assert "justify-content: flex-start" in settings_foot
    assert 'class="button button--small pg-reset-all"' in script
    assert '(settingsChanged() ? "" : " disabled")' in script
    assert "resetAll.disabled = !settingsChanged()" in script


def test_a_box_the_pipeline_does_not_have_is_refused():
    with pytest.raises(playground.SettingsError):
        _flowchart("output")


@pytest.mark.parametrize("temporal", TEMPORAL_CHOICES)
def test_every_value_a_flowchart_links_opens_a_setting_the_form_has(temporal):
    """A value drawn as a link leads to a setting, so the setting must exist."""
    pytest.importorskip("torch")
    keys = {f["key"] for f in playground.FIELDS}
    for stage in playground.STAGES:
        answer = _flowchart(stage, temporal=temporal)
        assert "error" not in answer, (stage, temporal, answer.get("error"))
        for step in _steps(answer):
            for part in step["parts"]:
                assert part["setting"] in keys | {None}, (stage, step["name"], part)


@pytest.mark.parametrize("settings", [
    *({"temporal": t} for t in TEMPORAL_CHOICES),
    {"temporal": "lstm", "shared_across_sensors": True, "temporal_out_features": 3},
    {"temporal": "gru", "gcn_bias": True, "gcn_hidden": [16, 8, 4], "fc_hidden": 20},
])
def test_the_two_boxes_of_the_network_show_every_learned_number_once(settings):
    """The temporal box and the graph box together show what the built network learns, no more and no less."""
    pytest.importorskip("torch")
    checked = playground.validate_settings(settings)
    shown = sum(step["learned"] or 0 for stage in ("temporal", "graph")
                for step in _steps(playground.stage_detail(stage, checked)))
    assert shown == playground.preview(checked)["parameters"]


@pytest.mark.parametrize("temporal", TEMPORAL_CHOICES)
def test_what_leaves_the_temporal_box_is_what_enters_the_graph_box(temporal):
    pytest.importorskip("torch")
    out = _flowchart("temporal", temporal=temporal, temporal_out_features=2)["steps"][-1]
    into = _flowchart("graph", temporal=temporal, temporal_out_features=2)["steps"][0]
    assert out["kind"] == into["kind"] == "edge"
    assert out["shape"] == into["shape"]


def test_a_window_the_summary_cannot_measure_change_over_is_refused_in_words():
    answer = _flowchart("temporal", temporal="window_summary", window_size=1)
    assert answer["steps"] == []
    assert "at least 2 cycles" in answer["error"]


def test_a_changed_value_never_wears_the_tag_of_the_default():
    """The paper prints a window of 30; a window of 40 is this reader's, and the step must not say 'Paper'."""
    default = _step(_flowchart("preprocessing"), "Cut into windows")
    assert default["source"] == {"kind": "paper", "ref": "Table 6", "text": ""}
    changed = _step(_flowchart("preprocessing", window_size=40), "Cut into windows")
    assert changed["source"] is None
    assert "\"Window length\" was changed from its default, 30 cycles (the paper's Table 6)." in changed["does"]


def test_a_count_of_one_takes_a_singular_noun_and_verb():
    """A window of one cycle and widths of one are allowed, so the words must still agree with them."""
    pytest.importorskip("torch")
    import re

    wrong = re.compile(r"\b1 (reading|number|cycle|pattern|pair|engine)s\b|\b1 reading (become|are)\b")
    for temporal in [*TEMPORAL_CHOICES[:4], "raw_window"]:
        for stage in ("temporal", "graph"):
            answer = _flowchart(stage, temporal=temporal, window_size=1, temporal_hidden=1,
                                gcn_hidden=[1, 1, 1], fc_hidden=1)
            for step in _steps(answer):
                for text in [step["does"]] + [part["text"] for part in step["parts"]]:
                    assert not wrong.search(text), (stage, temporal, text)


def test_one_network_shared_by_all_sensors_is_never_tagged_as_the_papers():
    """Section 2.2 gives each sensor its own network, so the shared one cites it only as the thing it departs from."""
    pytest.importorskip("torch")
    answer = _flowchart("temporal", shared_across_sensors=True)
    for name in ("Split by sensor", "One LSTM for all sensors"):
        assert _step(answer, name)["source"]["kind"] == "extension", name
    network = _step(answer, "One LSTM for all sensors")
    assert network["shape"].startswith("(B x 14, ")


def test_with_nothing_held_back_no_step_speaks_of_held_back_windows(raw_dir):
    answer = _flowchart("preprocessing", validation_fraction=0.0, early_stopping=False)
    names = [step["name"] for step in _steps(answer)]
    assert "Two sets of windows" not in names and "Three sets of windows" not in names
    assert [group["label"] for group in _step(answer, "Hold engines back")["outputs"]] == ["Training", "Test"]
    assert [group["label"] for group in _step(answer, "Cut into windows")["outputs"]] == ["Training", "Test"]
    assert "held-back" not in _step(answer, "Rescale each sensor")["does"]


def test_the_rescaled_graph_says_what_its_rows_sum_to_as_measured():
    """Symmetric rescaling leaves the rows of the complete graph summing to 1, and the words must not deny it."""
    pytest.importorskip("torch")
    complete = _step(_flowchart("graph", graph="complete", adjacency_normalization="symmetric"), "Rescale the graph")
    assert "no longer sum to 1" not in complete["does"] and "1/14 everywhere" in complete["does"]
    physics = _step(_flowchart("graph", adjacency_normalization="symmetric"), "Rescale the graph")
    assert "not to 1" in physics["does"]


def test_the_data_box_counts_the_rows_the_loader_reads(loaded):
    answer = _flowchart("data")
    assert _step(answer, "Training engines")["shape"] == f"({playground.design._count(len(loaded.train))}, 26)"
    assert _step(answer, "Test engines")["shape"] == f"({playground.design._count(len(loaded.test))}, 26)"
    assert not any(step.get("kind") == "evaluation" for step in answer["steps"])
    main = next(step for step in answer["steps"] if step.get("kind") == "group")
    assert not {"Test answers", "Attach test labels"}.intersection(step["name"] for step in main["steps"])
    assert [lane["name"] for lane in main["lanes"]] == ["Training data", "Test data"]
    assert [step["name"] for step in main["lanes"][1]["steps"]] == ["Test engines"]
    assert main["lanes"][1]["shape"] == _step(answer, "Test engines")["shape"]


@pytest.mark.parametrize("fraction,seed", [(0.2, 42), (0.0, 43)])
def test_the_preprocessing_groups_start_at_the_engine_split_and_keep_their_membership(raw_dir, fraction, seed):
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.data.windowing import split_units_by_id

    settings = playground.validate_settings({"window_size": 25, "validation_fraction": fraction, "seed": seed})
    prepared = run_preprocessing(PreprocessingConfig(**playground.preprocessing_settings(settings)), keep_frames=True)
    kept, held = split_units_by_id(prepared.raw.train.unit.unique(), validation_fraction=fraction, seed=seed)
    row_counts = [("Training", prepared.raw.train.unit.isin(kept).sum()),
                  *([("Validation", prepared.raw.train.unit.isin(held).sum())] if fraction else []),
                  ("Test", len(prepared.raw.test))]
    answer = playground.stage_detail("preprocessing", settings)
    for name, columns in (("Hold engines back", 27), ("Find operating conditions", 28), ("Rescale each sensor", 28)):
        step = _step(answer, name)
        assert [(o["label"], o["shape"]) for o in step["handoff_outputs"]] == [
            (label, f"({playground.design._count(n)}, {columns})") for label, n in row_counts]
    windows = _step(answer, "Cut into windows")
    groups = [("Training", prepared.train), *([("Validation", prepared.validation)] if fraction else []),
              ("Test", prepared.test)]
    assert [(o["label"], o["shape"], o["targets"]) for o in windows["handoff_outputs"]] == [
        (label, f"({playground.design._count(len(data))}, 14, 25)", f"({playground.design._count(len(data))},)")
        for label, data in groups]
    assert "no new split" in windows["does"]
    assert "Training → Procedure" in _step(answer, "Hold engines back")["summary"]
    assert not any(step.get("picture") == "sets" for step in answer["steps"])


def test_the_defaults_the_paper_states_are_the_ones_it_prints():
    """Table 4 and Table 6, read from the PDF on 2026-09-14."""
    defaults = {f["key"]: f["default"] for f in playground.FIELDS}
    assert defaults["gcn_hidden"] == [32, 64, 32]
    assert defaults["fc_hidden"] == 128
    assert (defaults["epochs"], defaults["batch_size"], defaults["window_size"]) == (200, 32, 30)
    assert (defaults["learning_rate"], defaults["final_learning_rate"]) == (1e-3, 1e-4)


@pytest.mark.parametrize(
    "payload",
    [
        {"no_such_setting": 1},
        {"subset": "FD005"},
        {"window_size": 0},
        {"seeds": 2.5},
        {"gcn_bias": "yes"},
        {"gcn_hidden": [32, 64]},
        {"learning_rate": float("nan")},
    ],
)
def test_a_bad_setting_is_refused_with_a_reason(payload):
    with pytest.raises(playground.SettingsError):
        playground.validate_settings(payload)


@pytest.mark.parametrize(
    ("temporal", "graph", "expected"),
    [
        ("lstm", "physics", "full"),
        ("last_cycle", "physics", "without_lstm"),
        ("lstm", "complete", "without_physics_prior"),
        ("last_cycle", "complete", None),
        ("lstm", "correlation", None),
        ("last_cycle", "correlation", None),
        ("gru", "physics", None),
        (None, "physics", None),
    ],
)
def test_a_configuration_is_set_beside_the_one_row_it_may_be(temporal, graph, expected):
    row, reason = playground.paper_row_for("FD001", temporal, graph)
    assert (row["key"] if row else None) == expected
    assert bool(reason) == (expected is None)
    # A stored run that recorded no graph-layer choice had them: the same row.
    assert playground.paper_row_for("FD001", temporal, graph, "gcn") == (row, reason)


@pytest.mark.parametrize(
    ("temporal", "graph", "expected"),
    [
        # With the graph layers out, the sensor graph is unused, so any graph
        # asked for gives the w/o GCN row with the LSTM stage.
        ("lstm", "physics", "without_gcn"),
        ("lstm", "complete", "without_gcn"),
        ("lstm", "correlation", "without_gcn"),
        ("last_cycle", "physics", None),
        ("last_cycle", "complete", None),
        ("gru", "physics", None),
        (None, "physics", None),
    ],
)
def test_a_network_without_graph_layers_is_set_beside_the_w_o_gcn_row_only_with_the_lstm(
        temporal, graph, expected):
    row, reason = playground.paper_row_for("FD001", temporal, graph, "none")
    assert (row["key"] if row else None) == expected
    assert bool(reason) == (expected is None)
    if expected:
        assert (row["label"], row["source"], row["rmse"]) == ("ESTGNN w/o GCN", "Table 11", 16.42)
    elif temporal is not None:
        assert reason.startswith("Two parts are removed at once")


def test_the_four_combinations_of_the_two_parts_the_overview_takes_out():
    """LSTM in or out, graph layers in or out, on the paper's graph: three rows and one refusal."""
    keys = {}
    for temporal in ("lstm", "last_cycle"):
        for spatial in ("gcn", "none"):
            row, _ = playground.paper_row_for("FD001", temporal, "physics", spatial)
            keys[(temporal, spatial)] = row["key"] if row else None
    assert keys == {("lstm", "gcn"): "full", ("lstm", "none"): "without_gcn",
                    ("last_cycle", "gcn"): "without_lstm", ("last_cycle", "none"): None}


def _cautions(**changes) -> list[dict]:
    return playground.caveats(playground.validate_settings(changes))


def test_fewer_than_five_seeds_is_marked_exploratory_in_words():
    [one] = [note for note in _cautions(seeds=1) if "exploratory" in note["text"]]
    assert one["text"] == ("1 seed is exploratory: quote nothing from it. A result is a mean "
                           "over at least five seeds, with its spread.")
    assert (one["ref"], one["where"]) == ("", ["design", "run"])
    assert any(note["text"].startswith("2 seeds is exploratory:") for note in _cautions(seeds=2))
    assert not any("exploratory" in note["text"] for note in _cautions())


def test_only_the_two_conflicts_a_reader_may_not_see_are_shown_beside_train():
    """Every caution qualifies a stored run; seeds and a plateau with nothing held back also flag the design."""
    combinations = [
        {}, {"seeds": 2}, {"temporal": "last_cycle"}, {"temporal": "gru", "shared_across_sensors": True},
        {"graph": "complete", "fc_hidden": 64}, {"graph": "correlation", "fixed_split": True},
        {"validation_fraction": 0.0, "lr_schedule": "plateau"},
    ]
    shown_beside_train = set()
    for changes in combinations:
        for note in _cautions(**changes):
            assert "run" in note["where"], note
            if "design" in note["where"]:
                shown_beside_train.add(note["ref"] or note["text"].split(" is ")[0])
    assert shown_beside_train == {"2 seeds", "A-17"}


def test_the_cautions_are_worded_as_decided():
    [best] = [note for note in _cautions() if note["ref"] == "F-16"]
    assert best["text"] == ("The paper does not describe selecting an earlier pass by validation error. "
                            "This project's measured LSTM comparison depends on whether that selection is used.")
    # Written with no window length in it, so a 20-cycle window reads true too.
    [last] = [note for note in _cautions(temporal="last_cycle", window_size=20) if note["ref"] == "D-6"]
    assert not any(character.isdigit() for character in last["text"])
    # The recurrent stage's sizes are this project's choice for the GRU as much as the LSTM.
    assert [note["ref"] for note in _cautions(temporal="gru")].count("A-26") == 1
    # The run's page already says why no published row applies.
    for stage in ("gru", "mean_window", "window_summary", "raw_window"):
        assert not any("not in the paper" in note["text"] for note in _cautions(temporal=stage))
    assert not any("seed(s)" in note["text"] for note in _cautions(seeds=3))


def test_left_out_engines_are_counted_from_the_files_not_guessed_from_the_name(raw_dir: Path):
    """FD002 is not always short of engines, and FD001 is not always whole: the window decides."""
    import pandas as pd

    if not (raw_dir / "test_FD002.txt").is_file():
        pytest.skip(f"test_FD002.txt is not in {raw_dir}.")

    def left_out(subset: str, window: int) -> list[dict]:
        return [note for note in _cautions(subset=subset, window_size=window) if note["ref"] == "A-23"]

    assert left_out("FD002", 20) == []
    # Counted here with pandas alone, apart from the package's own reader.
    frame = pd.read_csv(raw_dir / "test_FD001.txt", sep=r"\s+", header=None)
    lengths = frame.groupby(0)[1].max()
    short = int((lengths < 40).sum())
    assert short > 0
    [note] = left_out("FD001", 40)
    assert note["text"].startswith(f"{short} of {len(lengths)} test engines ")
    assert note["text"].endswith("so the score cannot be set beside Table 7.")
    assert note["where"] == ["run"]


def test_the_preview_carries_only_what_the_design_tab_draws():
    import importlib.util

    answer = playground.preview(playground.validate_settings({}))
    assert "command" not in answer and "caveats" not in answer
    # The row of the paper is shown for a stored run, not while designing one.
    assert "paper_row" not in answer and "no_row_reason" not in answer
    assert set(answer["dataset"]) == {"table5", "counted", "files"}
    if answer["dataset"]["files"]:
        files = answer["dataset"]["files"]
        assert files["train_engines"] == answer["dataset"]["table5"]["train_engines"]
        assert files["test_engines"] == answer["dataset"]["table5"]["test_engines"]
        assert files["labels"] == files["test_engines"]
        assert files["train_history"]["shortest"] <= files["train_history"]["median"] <= files["train_history"]["longest"]
        assert files["test_history"]["shortest"] <= files["test_history"]["median"] <= files["test_history"]["longest"]
        sample = files["sample"]
        assert sample["source"] == "train_FD001.txt"
        assert len(sample["rows"]) == 10
        assert len({row["unit"] for row in sample["rows"]}) == 1
        assert [row["cycle"] for row in sample["rows"]] == list(range(1, 11))
        assert [column["key"] for column in sample["columns"]] == list(sample["rows"][0])
        from thesis_gnn_rul_xai.data.sensors import RAW_COLUMNS

        assert [column["key"] for column in sample["columns"]] == RAW_COLUMNS
    assert answer["design_cautions"] == []
    assert answer["torch"] is (importlib.util.find_spec("torch") is not None)
    if answer["torch"]:
        assert answer["parameters"] > 0 and answer["layers"]
        assert "network_error" not in answer
    few = playground.preview(playground.validate_settings({"seeds": 2}))
    assert [note["text"].split(":")[0] for note in few["design_cautions"]] == [
        "2 seeds is exploratory"]


def test_without_pytorch_the_preview_says_so_by_a_flag_not_as_a_network_error(monkeypatch):
    """The page disables Train on the flag; a network error is kept for settings the network refuses."""
    monkeypatch.setitem(sys.modules, "torch", None)  # "import torch" now fails
    # Imported already, the network's modules would not import PyTorch again.
    for name in [n for n in sys.modules if n.startswith("thesis_gnn_rul_xai.models")]:
        monkeypatch.delitem(sys.modules, name)
    answer = playground.preview(playground.validate_settings({}))
    assert answer["torch"] is False
    assert not {"network_error", "layers", "parameters"} & set(answer)


def test_the_command_is_a_list_of_arguments_never_a_shell_string():
    settings = playground.validate_settings({"early_stopping": False, "temporal": "last_cycle"})
    arguments = playground.train_arguments(settings, Path("config.yaml"), Path("p.jsonl"))
    assert arguments[:3] == [sys.executable, "-u", "scripts/train.py"]
    assert "--no-early-stopping" in arguments
    assert "--temporal-hidden" not in arguments  # the removed stage has no size
    shown = playground.train_arguments(settings, Path("config.yaml"), None, display=True)
    assert shown[:2] == ["python", "scripts/train.py"]
    assert "--progress" not in shown


def test_the_graph_layers_can_be_left_out_and_say_where_that_choice_comes_from():
    spec = next(f for f in playground.FIELDS if f["key"] == "spatial")
    assert (spec["group"], spec["kind"], spec["default"]) == ("graph", "choice", "gcn")
    choices = {c["value"]: c for c in spec["choices"]}
    assert set(choices) == {"gcn", "none"}
    assert choices["gcn"]["source"]["kind"] == "paper"
    assert (choices["none"]["source"]["kind"], choices["none"]["source"]["ref"]) == ("assumption", "A-42")
    assert playground.read_more("spatial=none") == "#/entry/A-42"
    assert playground.validate_settings({})["spatial"] == "gcn"
    assert playground.validate_settings({"spatial": "none"})["spatial"] == "none"
    with pytest.raises(playground.SettingsError, match="Graph layers in the network"):
        playground.validate_settings({"spatial": "gat"})


def test_the_default_command_is_unchanged_and_no_graph_layers_add_one_flag():
    """The default network's command names no graph-layer choice; leaving them out
    passes --spatial none and drops the widths and offsets of layers that do not exist."""
    default = playground.train_arguments(playground.validate_settings({}), Path("config.yaml"), None, display=True)
    assert default == [
        "python", "scripts/train.py", "--config", "config.yaml", "--seeds", "5", "--temporal", "lstm",
        "--temporal-hidden", "32", "--temporal-out-features", "1", "--gcn-hidden", "32", "64", "32",
        "--fc-hidden", "128", "--epochs", "200", "--batch-size", "32", "--learning-rate", "0.001",
        "--final-learning-rate", "0.0001", "--lr-schedule", "step", "--early-stopping-patience", "20",
        "--device", "auto"]
    settings = playground.validate_settings({"spatial": "none", "gcn_bias": True, "gcn_hidden": [8, 8, 8]})
    shown = playground.train_arguments(settings, Path("config.yaml"), None, display=True)
    start = default.index("--gcn-hidden")
    assert shown == [*default[:start], "--spatial", "none", *default[start + 4:]]


def test_a_network_without_graph_layers_is_cautioned_and_named_in_its_job():
    settings = playground.validate_settings({"spatial": "none", "graph": "complete", "gcn_hidden": [8, 8, 8]})
    refs = [note["ref"] for note in playground.caveats(settings)]
    assert "A-42" in refs
    # The unused graph and widths raise none of their own cautions.
    assert "D-2" not in refs and "Table 4" not in refs
    assert "w/o GCN" in playground.job_label(settings)
    assert "graph" not in playground.job_label(settings)
    assert "physics graph" in playground.job_label(playground.validate_settings({}))


def test_the_graph_box_without_graph_layers_flattens_the_temporal_stage_into_the_first_dense_layer():
    pytest.importorskip("torch")
    settings = playground.validate_settings({"spatial": "none", "gcn_hidden": [8, 8, 8], "gcn_bias": True})
    steps = playground.stage_detail("graph", settings)["steps"]
    names = [step["name"] for step in steps]
    assert not any(name.startswith(("Graph layer", "Sensor graph", "Rescale the graph")) for name in names)
    assert all(step.get("kind") != "branch" for step in steps)
    flatten = next(step for step in steps if step["name"] == "Flatten")
    assert (flatten["shape"], flatten["source"]["ref"]) == ("(B, 14)", "A-42")
    assert [part["text"] for part in flatten["parts"]] == ["14 x 1 = 14", "no graph layers"]
    temporal = playground.stage_detail("temporal", settings)["steps"]
    assert temporal[-1]["name"] == "To the fully connected layers, with no graph layers"
    preview = playground.preview(settings)
    assert [row["layer"] for row in preview["layers"]] == ["Input", "Flatten", "FC-1", "FC-2"]


def _stored_run(tmp_path: Path, name: str, model: dict | None) -> Path:
    """A finished run's record, the way train.py writes one, with or without the graph-layer choice."""
    run = tmp_path / "reports" / name
    (run / "seed_42").mkdir(parents=True)
    (run / "results.csv").write_text("seed,rmse\n42,12.0\n", encoding="utf-8")
    (run / "seed_42" / "model.pt").write_bytes(b"")
    summary = {"mean": {"rmse": 12.0}, "std": {"rmse": None}, "subset": "FD001", "seeds": 1,
               "temporal_stage": "lstm", "graph": "physics", "finished_utc": "20261113T000000Z"}
    if model is not None:
        summary["model"] = model
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def test_a_stored_run_is_read_as_having_graph_layers_unless_it_says_otherwise(tmp_path: Path):
    _stored_run(tmp_path, "old", None)
    _stored_run(tmp_path, "full", {"temporal": "lstm", "spatial": "gcn"})
    _stored_run(tmp_path, "nogcn", {"temporal": "lstm", "spatial": "none"})
    runs = {run["name"]: run for run in playground.list_runs(tmp_path)}
    assert {name: run["spatial"] for name, run in runs.items()} == {"old": "gcn", "full": "gcn", "nogcn": "none"}
    assert runs["old"]["paper_row"]["key"] == runs["full"]["paper_row"]["key"] == "full"
    assert runs["nogcn"]["paper_row"]["key"] == "without_gcn"
    assert playground.run_detail(tmp_path, "nogcn")["spatial"] == "none"


def test_a_network_without_graph_layers_is_offered_no_walks(tmp_path: Path):
    """Walks follow connections through the graph layers; the other two methods vary the input only."""
    _stored_run(tmp_path, "nogcn", {"temporal": "lstm", "spatial": "none"})
    _stored_run(tmp_path, "old", None)
    base = {"run": "nogcn", "seed": 42, "scope": "engine", "engine": 7, "every": 1}
    for method in ("shap", "occlusion"):
        assert playground.validate_explain({**base, "method": method}, tmp_path)["method"] == method
    with pytest.raises(playground.SettingsError, match="no graph layers"):
        playground.validate_explain({**base, "method": "walks"}, tmp_path)
    assert playground.validate_explain({**base, "run": "old", "method": "walks"}, tmp_path)["gamma"] == "authors"
    # A stored explanation of it carries no connections to draw.
    folder = tmp_path / "reports" / "nogcn" / "explanations" / "shap_engine"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"method": "shap", "seed": 42}), encoding="utf-8")
    detail = playground.explanation_detail(tmp_path, "nogcn", "shap_engine")
    assert (detail["graph_layers"], detail["graph_edges"]) == (False, [])
    # The page leaves walks out of the method list for such a run, and says why.
    script = (ROOT / "src/thesis_gnn_rul_xai/docsite/assets/playground.js").read_text(encoding="utf-8")
    assert 'return layered || option.value !== "walks";' in script
    assert "Walks through the sensor graph are not offered" in script
    # Its explanation is drawn on the engine panel, which is handed the stored connections: none for such a run.
    assert "edges: data.graph_edges === undefined ? null : data.graph_edges" in script


# -- the site --------------------------------------------------------------------


def test_the_catalog_lists_the_playground():
    paths = [doc.path for doc in catalog.build_catalog(ROOT).by_path.values()]
    assert "explore/playground" in paths


def test_the_page_loads_the_playground_before_the_script_that_routes_to_it():
    shell = (ROOT / "src/thesis_gnn_rul_xai/docsite/assets/shell.html").read_text(encoding="utf-8")
    assert shell.index("playground.js") < shell.index("app.js")


@pytest.fixture()
def site(tmp_path: Path):
    """A server on a free port, serving an empty project, stopped afterwards."""
    handler = type("TestHandler", (server.DocsHandler,), {"root": tmp_path})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url: str, body: dict, origin: str | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    if origin:
        request.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read())


def test_another_web_page_cannot_start_work(site: str):
    status, answer = _post(site + "/api/playground/cancel", {"id": "x"}, origin="http://example.com")
    assert status == 403
    assert "did not come from this site" in answer["error"]


def test_the_site_refuses_a_box_it_has_no_flowchart_for(site: str):
    status, answer = _post(site + "/api/playground/stage", {"stage": "output", "settings": {}}, origin=site)
    assert status == 400
    assert "stage must be one of" in answer["error"]


def test_the_site_explains_a_refused_setting(site: str):
    status, answer = _post(site + "/api/playground/preview", {"seeds": 0}, origin=site)
    assert status == 400
    assert "Seeds" in answer["error"]


def test_the_server_names_a_job_the_page_asks_for_whatever_name_the_request_carries(
    site: str, tmp_path: Path
):
    """The page sends no name; an older page's name is ignored, so a job has one name only."""
    run = tmp_path / "reports" / "somerun"
    (run / "seed_42").mkdir(parents=True)
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "seed_42" / "model.pt").write_bytes(b"")
    request = {"run": "somerun", "seed": 42, "method": "occlusion", "scope": "engine",
               "engine": 7, "every": 1, "label": "a name the page made up"}
    status, job = _post(site + "/api/playground/explain", request, origin=site)
    try:
        assert status == 200, job
        assert job["id"]
        assert job["label"] == "Occlusion · engine 7 · seed 42"
    finally:
        # The command cannot start in this empty folder, so it ends at once.
        manager = server.job_manager(tmp_path)
        _wait(manager, job["id"])
        manager.stop()


def test_walks_are_a_method_the_server_accepts_and_names(site: str, tmp_path: Path):
    run = tmp_path / "reports" / "somerun"
    (run / "seed_42").mkdir(parents=True)
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "seed_42" / "model.pt").write_bytes(b"")
    request = {"run": "somerun", "seed": 42, "method": "walks", "scope": "engine",
               "engine": 7, "every": 1}
    status, job = _post(site + "/api/playground/explain", request, origin=site)
    try:
        assert status == 200, job
        assert job["label"] == "Walks through the sensor graph · engine 7 · seed 42"
    finally:
        manager = server.job_manager(tmp_path)
        _wait(manager, job["id"])
        manager.stop()
    with pytest.raises(playground.SettingsError, match="'shap', 'occlusion' or 'walks'"):
        playground.validate_explain({**request, "method": "lrp"}, tmp_path)


def test_the_page_and_the_server_name_the_same_explanation_methods():
    """The page's METHODS and the server's names are one vocabulary."""
    script = (ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets" / "playground.js").read_text(
        encoding="utf-8")
    line = next(x for x in script.splitlines() if x.strip().startswith("var METHODS = {"))
    for key, name in playground.settings._METHOD_NAMES.items():
        assert f'{key}: "{name}"' in line, (key, name)


def test_explanation_charts_share_one_full_width_tabbed_area():
    """Every chart is chosen from one accessible tab row, and the deck is not capped at the document width."""
    import re

    assets = ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets"
    script = (assets / "playground.js").read_text(encoding="utf-8")
    css = (assets / "app.css").read_text(encoding="utf-8")
    deck = re.search(r"\.pg-explanation-charts\s*\{([^}]*)\}", css).group(1)

    assert 'role="tablist" aria-label="Explanation charts"' in script
    assert 'role="tabpanel" aria-labelledby="pg-chart-tab-' in script
    assert 'data-action="chart-tab"' in script
    assert 'class="pg-charts pg-explanation-charts"' in script
    assert "width: 100%" in deck
    assert "max-width: none" in deck
    explanation_width = script.split("function explanationChartWidth", 1)[1].split("function chartHeight", 1)[0]
    assert "Math.min(panel.clientWidth, 1080)" not in explanation_width


def test_project_added_charts_state_their_origin_once_and_explain_their_use():
    """Method origin belongs in the guide; every tab of a walk explanation opens with the question it answers."""
    assets = ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets"
    script = (assets / "playground.js").read_text(encoding="utf-8")
    engine_view = (assets / "engine-view.js").read_text(encoding="utf-8")
    walks = script.split("function drawWalks", 1)[1].split("var FLIP_STYLES", 1)[0]

    assert "<strong>Not in the paper.</strong>" not in script
    assert "<strong>Not in the paper." not in engine_view
    assert walks.count("Walk explanations are added by this project") == 1
    # The tabs are named for what they hold, and each begins with its question.
    for label in ('label: "Score breakdown"', 'label: "Largest walks"', 'label: "Explanation check"'):
        assert label in walks, label
    for question in ("What is the score made of?", "Does the explanation agree with the network?",
                     "How much passed along each connection, on average?"):
        assert 'walkTabHead("' + question + '"' in walks, question
    assert 'walkTabHead("Which walks carry the most?"' in script
    # The chosen score as a sum comes before the same parts for every cycle or engine.
    assert walks.index("scoreSum(data, e.index)") < walks.index("walkParts(data, e.index, W)")
    # What the tabs were called, and the words the author could not interpret, are gone.
    for gone in ('"Score parts"', "Authors' check\"", "model-added part", "routes within one group", "How to read this chart."):
        assert gone not in script, gone


def test_engine_drawing_is_compact_inside_the_full_width_chart():
    """The full-width panel must not enlarge the engine until its sensor details fall far below it."""
    import re

    css = (ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets" / "app.css").read_text(
        encoding="utf-8")
    drawing = re.search(r"\.ev-drawing\s*\{([^}]*)\}", css).group(1)

    assert "width: 100%" in drawing
    assert "max-width: 1000px" in drawing
    assert "margin: 0 auto" in drawing


def test_test_engine_walk_parts_are_not_drawn_as_an_ordered_engine_sequence():
    """Engine number is an identifier; one row of dots per part replaces its scatter plot, with the chosen engine ringed."""
    script = (ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets" / "playground.js").read_text(
        encoding="utf-8")
    section = script.split("function walkParts", 1)[1].split("function flowGraph", 1)[0]
    assert "walkPartDistribution(rows, W, at, parts)" in section
    assert "the chosen one" in section
    assert "middle half of the engines" in section
    assert 'xlabel: "test engine"' not in section


# -- jobs ----------------------------------------------------------------------


def _wait(manager: playground.JobManager, job_id: str, seconds: float = 60) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = manager.get(job_id)
        if job is not None and job.status in ("done", "failed", "cancelled"):
            return job.to_dict()
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish in {seconds}s")


def test_a_job_runs_and_what_it_printed_is_kept(tmp_path: Path):
    manager = playground.JobManager(tmp_path)
    job = manager.submit_command("check", "print", [sys.executable, "-c", "print('ran')"])
    finished = _wait(manager, job.id)
    assert finished["status"] == "done"
    assert "ran" in (job.folder / "output.txt").read_text(encoding="utf-8")
    manager.stop()


def test_a_running_job_can_be_stopped(tmp_path: Path):
    manager = playground.JobManager(tmp_path)
    job = manager.submit_command("check", "sleep", [sys.executable, "-c", "import time; time.sleep(60)"])
    deadline = time.time() + 20
    while manager.get(job.id).status != "running" and time.time() < deadline:
        time.sleep(0.1)
    manager.cancel(job.id)
    assert _wait(manager, job.id, 20)["status"] == "cancelled"
    manager.stop()


def test_a_finished_job_can_be_removed_and_a_running_one_cannot(tmp_path: Path):
    manager = playground.JobManager(tmp_path)
    done = manager.submit_command("check", "print", [sys.executable, "-c", "print('ran')"])
    _wait(manager, done.id)
    assert json.loads((done.folder / "job.json").read_text(encoding="utf-8"))["status"] == "done"
    running = manager.submit_command("check", "sleep", [sys.executable, "-c", "import time; time.sleep(60)"])
    deadline = time.time() + 20
    while manager.get(running.id).status != "running" and time.time() < deadline:
        time.sleep(0.1)
    with pytest.raises(playground.SettingsError):
        manager.forget(running.id)
    assert manager.forget(done.id) == [done.id]
    assert not done.folder.exists()
    assert manager.get(done.id) is None
    manager.cancel(running.id)
    _wait(manager, running.id, 20)
    manager.stop()


def test_a_job_another_server_is_still_running_is_not_reported_as_ended(tmp_path: Path):
    """Reported ended, its folder would be offered to "remove all ended" and deleted under it."""
    import os

    folder = tmp_path / "reports" / ".playground" / "20260916T120000_train_abcdef"
    folder.mkdir(parents=True)
    (folder / "progress.jsonl").write_text('{"event": "start", "seeds": 1}\n', encoding="utf-8")
    (folder / "job.json").write_text(json.dumps({
        "id": folder.name, "kind": "train", "label": "still going", "command": "x",
        "created": 1.0, "status": "running", "started": 1.0, "finished": None,
        "returncode": None, "pid": os.getpid(),
    }), encoding="utf-8")

    manager = playground.JobManager(tmp_path)
    assert {item["id"]: item["status"] for item in manager.snapshot()}[folder.name] == "running"
    with pytest.raises(playground.SettingsError):
        manager.forget(folder.name)
    assert manager.forget(all_finished=True) == []
    assert folder.is_dir()

    # The same folder, once nothing is running it, is an interrupted job again.
    (folder / "job.json").write_text(json.dumps({
        "id": folder.name, "kind": "train", "label": "stopped with its server",
        "command": "x", "created": 1.0, "status": "running", "pid": 0,
    }), encoding="utf-8")
    assert {item["id"]: item["status"] for item in manager.snapshot()}[folder.name] == "interrupted"
    assert manager.forget(all_finished=True) == [folder.name]
    assert not folder.exists()


@pytest.mark.parametrize(
    "name", ["C:", "C:x", "C:Windows", "../secret", "..", "", ".hidden", "sub/dir"]
)
def test_a_job_name_that_points_outside_the_jobs_folder_is_refused(tmp_path: Path, name: str):
    """Joining the name onto the jobs folder must stay inside it: "C:x" carries a drive and does not."""
    manager = playground.JobManager(tmp_path)
    with pytest.raises(playground.SettingsError):
        manager.forget(name)


def test_a_job_record_with_a_damaged_process_number_does_not_break_the_listing(tmp_path: Path):
    """Every job on the page comes from this listing, so one bad field must not raise."""
    folder = tmp_path / "reports" / ".playground" / "20260916T130000_train_badpid"
    folder.mkdir(parents=True)
    (folder / "progress.jsonl").write_text('{"event": "start", "seeds": 1}\n', encoding="utf-8")
    (folder / "job.json").write_text(json.dumps({
        "id": folder.name, "kind": "train", "label": "damaged", "status": "running",
        "created": 1.0, "pid": "not a number",
    }), encoding="utf-8")

    listed = {item["id"]: item for item in playground.JobManager(tmp_path).snapshot()}
    assert listed[folder.name]["status"] == "interrupted"


@pytest.mark.parametrize("name", ["C:", "C:Windows", "../secret", "..", "", ".hidden", "sub/.hidden"])
def test_a_run_name_that_points_outside_reports_is_refused(tmp_path: Path, name: str):
    """The same hazard as a job name: a name carrying a drive throws the base away."""
    with pytest.raises(playground.SettingsError):
        playground.run_detail(tmp_path, name)


def test_an_explanation_name_that_points_outside_its_run_is_refused(tmp_path: Path):
    """An explanation is read from inside its run's folder and nowhere else."""
    run = tmp_path / "reports" / "somerun"
    run.mkdir(parents=True)
    (run / "summary.json").write_text("{}", encoding="utf-8")
    for bad in ("C:", "C:Windows", "..", "sub/dir", "", ".hidden"):
        with pytest.raises(playground.SettingsError):
            playground.explanation_detail(tmp_path, "somerun", bad)


def test_nested_training_runs_are_listed_by_their_path_below_reports(tmp_path: Path):
    """Grouped sensitivity runs remain distinct and can be opened from Run History."""
    reports = tmp_path / "reports"
    for dataset in ("FD002", "FD003"):
        run = reports / "sensitivity_study" / "runs" / dataset / "reference"
        run.mkdir(parents=True)
        (run / "results.csv").write_text("seed,rmse\n42,12.0\n", encoding="utf-8")
        (run / "summary.json").write_text(json.dumps({
            "mean": {"rmse": 12.0}, "std": {"rmse": 0.0}, "subset": dataset,
            "seeds": 1, "finished_utc": f"2026-09-{dataset[-1]}T00:00:00Z",
        }), encoding="utf-8")

    names = {run["name"] for run in playground.list_runs(tmp_path)}
    assert names == {
        "sensitivity_study/runs/FD002/reference",
        "sensitivity_study/runs/FD003/reference",
    }
    detail = playground.run_detail(tmp_path, "sensitivity_study/runs/FD002/reference")
    assert detail["subset"] == "FD002"
    assert detail["name"] == "sensitivity_study/runs/FD002/reference"
    assert detail["reported_weights"] is None
    assert detail["preprocessing"] == detail["recorded_model"] == {}
    assert not any(note["ref"] == "A-02" for note in detail["caveats"])


def test_saved_settings_render_all_fields_without_using_current_defaults():
    """Recorded false, zero, absent and seed-specific settings stay distinct in the UI."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the saved-settings view.")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const context = {window: {}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {runConfiguration, runInterpretation, setup: function (helpers) { T = helpers; }};`), context);
const P = context.window.test;
const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
P.setup({escapeHtml});
const record = {
  preprocessing: {validation_fraction: 0, clip_test_labels: false, smoothing_alpha: null,
    notes: '<script>unsafe</script>', future_preparation_setting: 'retained'},
  model: {temporal: 'last_cycle', shared_across_sensors: false, gcn_bias: false, gcn_hidden: [8, 4]},
  recorded_summary: {training: {batch_size: 128}, seeds: 2, split: 'fixed'},
  per_seed: {'42': {training: {batch_size: 16, seed: 42, gradient_clip: null}},
             '43': {training: {batch_size: 64, seed: 43, optimizer: 'adam'}}},
  results: [{seed: 42}, {seed: 43}]
};
const html = P.runConfiguration(record);
function setting(key) {
  return html.split('data-setting="' + key + '">')[1].split('</dd>')[0];
}
assert.match(setting('validation_fraction'), /<dd>0/);
assert.match(setting('clip_test_labels'), /<dd>No/);
assert.match(setting('smoothing_alpha'), /<dd>Off/);
assert.match(setting('gcn_bias'), /<dd>No/);
assert.match(setting('gcn_hidden'), /\[8,4\]/);
assert.match(setting('future_preparation_setting'), /retained/);
assert.match(setting('window_size'), /Not recorded/);
assert.match(setting('batch_size'), /Seed 42:<\/span> 16/);
assert.match(setting('batch_size'), /Seed 43:<\/span> 64/);
assert.ok(!setting('batch_size').includes('128'));
assert.match(setting('optimizer'), /Seed 42:<\/span>.*Not recorded/);
assert.match(setting('optimizer'), /Seed 43:<\/span> Adam/);
assert.ok(html.includes('&lt;script&gt;unsafe&lt;/script&gt;') && !html.includes('<script>'));
assert.ok(html.includes('unused by this temporal route'));
const legacy = P.runConfiguration({training: {batch_size: 128}, results: [{seed: 42}]});
assert.match(legacy, /<dd>128.*\(run summary\)/);
assert.ok(P.runConfiguration({}).includes('Not recorded'));
const summaryOnly = P.runConfiguration({recorded_summary: {subset: 'FD003', temporal_stage: 'gru', temporal_hidden: 16}});
assert.match(summaryOnly, /<dd>FD003.*\(run summary\)/);
assert.match(summaryOnly, /<dd>GRU.*\(run summary\)/);
const notes = P.runInterpretation([{ref: 'A-26', text: '<test>', level: 'info'},
  {ref: 'F-16', text: 'Saved choice', level: 'warn'}], '<p>Comparison</p>');
assert.ok(notes.includes('How to interpret this run') && notes.includes('Temporal choices'));
assert.ok(notes.includes('Choosing the reported weights') && notes.includes('#/entry/F-16'));
assert.ok(notes.includes('&lt;test&gt;') && !notes.includes('<test>'));
"""
    rendered = subprocess.run([node, "-e", script], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    assert rendered.returncode == 0, rendered.stderr


def test_jobs_an_earlier_server_left_on_disk_are_listed_from_their_folders(tmp_path: Path):
    import calendar

    folder = tmp_path / "reports" / ".playground" / "20260914T125018_train_046c54"
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text(json.dumps(
        {"subset": "FD001", "temporal": "last_cycle", "graph": "physics",
         "validation_fraction": 0.2, "seeds": 2}), encoding="utf-8")
    (folder / "progress.jsonl").write_text(
        '{"event": "start", "seeds": 2}\n{"event": "done", "directory": "some_run"}\n', encoding="utf-8")
    # One that never wrote its end is reported as interrupted, not as done.
    other = tmp_path / "reports" / ".playground" / "20260914T132855_train_15d73a"
    other.mkdir()
    (other / "progress.jsonl").write_text('{"event": "start", "seeds": 5}\n', encoding="utf-8")

    listed = {item["id"]: item for item in playground.JobManager(tmp_path).snapshot()}
    finished = listed["20260914T125018_train_046c54"]
    assert finished["status"] == "done"
    assert finished["from_disk"] is True
    assert finished["label"] == "FD001 · w/o LSTM · physics graph · best pass · 2 seeds"
    assert finished["progress"]["result"] == {"directory": "some_run"}
    assert finished["created"] == calendar.timegm((2026, 9, 14, 12, 50, 18, 0, 0, 0))
    assert listed["20260914T132855_train_15d73a"]["status"] == "interrupted"


def test_a_job_is_named_in_the_same_words_live_and_read_back_from_its_folder(tmp_path: Path):
    """One builder names every job, so a later server lists a folder under the name the page first showed."""
    manager = playground.JobManager(tmp_path)
    # Neither command can start in an empty folder, which has no scripts/,
    # so both end at once as failed and leave only their folders behind.
    train = manager.submit_training(
        playground.validate_settings({"seeds": 2, "adjacency_normalization": "symmetric"}))
    explain = manager.submit_explanation(
        {"run": "somerun", "seed": 42, "method": "shap", "scope": "engine", "engine": None, "every": 1})
    for job in (train, explain):
        _wait(manager, job.id)
    manager.stop()

    def names(listing: list[dict]) -> dict[str, str]:
        return {item["id"]: item["label"] for item in listing}

    live = names(manager.snapshot())
    assert live[train.id] == "FD001 · LSTM 32 · physics · symmetric graph · best pass · 2 seeds"
    assert live[explain.id] == "Exact Shapley values · longest engine · seed 42"
    assert names(playground.JobManager(tmp_path).snapshot()) == live

    # Finished, an explanation's own summary names the engine the script chose.
    stored = tmp_path / "reports" / "somerun" / "explanations" / "shap_engine_unit5_seed42"
    stored.mkdir(parents=True)
    (stored / "summary.json").write_text(json.dumps(
        {"method": "shap", "scope": "engine", "unit": 5, "seed": 42}), encoding="utf-8")
    (explain.folder / "progress.jsonl").write_text(json.dumps(
        {"event": "done", "directory": stored.name, "run": "somerun"}) + "\n", encoding="utf-8")
    live = names(manager.snapshot())
    assert live[explain.id] == "Exact Shapley values · engine 5 · seed 42"
    assert names(playground.JobManager(tmp_path).snapshot()) == live


# -- end to end, on the real data ----------------------------------------------------


def _train(root: Path, name: str, settings: dict) -> Path:
    """Run the command the playground would run, into a folder of the test's own."""
    config = root / f"{name}.yaml"
    config.write_text(json.dumps(playground.preprocessing_settings(settings)), encoding="utf-8")
    output = root / "reports" / name
    arguments = playground.train_arguments(settings, config, root / f"{name}.jsonl")
    completed = subprocess.run(
        [*arguments, "--output", str(output), "--quiet"], cwd=ROOT,
        capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    return output


def test_a_playground_run_stores_everything_the_page_reads_back(raw_dir, tmp_path: Path):
    pytest.importorskip("torch")
    quick = {"epochs": 1, "seeds": 1, "validation_fraction": 0.2}
    first = _train(tmp_path, "first", playground.validate_settings({**quick, "temporal": "last_cycle"}))
    _train(tmp_path, "second", playground.validate_settings({**quick, "temporal": "mean_window"}))

    for kept in ("model.pt", "training_history.csv", "training_config.yaml", "test_predictions.csv"):
        assert (first / "seed_42" / kept).is_file(), kept
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["reported_weights"] == "best_validation_epoch"
    assert summary["model"]["gcn_hidden"] == [32, 64, 32]
    assert [json.loads(line)["event"] for line in (tmp_path / "first.jsonl").read_text().splitlines()][-1] == "done"

    runs = playground.list_runs(tmp_path)
    assert {run["name"] for run in runs} == {"first", "second"}
    detail = playground.run_detail(tmp_path, "first")
    assert len(detail["per_seed"]["42"]["predictions"]) == 100
    assert len(detail["per_seed"]["42"]["history"]) == 1
    assert detail["recorded_summary"] == summary
    assert detail["recorded_model"] == summary["model"]
    saved_training = playground.runs._read_yaml(first / "seed_42" / "training_config.yaml")
    assert detail["per_seed"]["42"]["training"] == saved_training
    assert saved_training["optimizer"] == "adam"
    assert saved_training["gradient_clip"] is None

    comparison = playground.compare_runs(tmp_path, "first", "second")
    assert {"setting": "model.temporal", "first": "last_cycle", "second": "mean_window"} in comparison["differences"]
    # One shared seed can be paired but not tested: a spread needs two.
    assert comparison["shared_seeds"] == [42]
    assert comparison["measures"]["rmse"]["p"] is None

    explained = subprocess.run(
        [sys.executable, "scripts/explain.py", "--run", str(first), "--method", "shap",
         "--scope", "engine", "--every", "60", "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert explained.returncode == 0, explained.stdout[-2000:] + explained.stderr[-2000:]
    stored = playground.run_detail(tmp_path, "first")["explanations"]
    assert len(stored) == 1
    values = playground.explanation_detail(tmp_path, "first", stored[0]["name"])
    # Equation 14: a window's Shapley values add up to its distance from the baseline.
    assert values["summary"]["sum_gap_max"] < 1e-3
    assert len(values["values"]) == values["summary"]["windows"]
    # The paper's Fig. 9 colours each value by the sensor's own reading, so
    # every window stores the reading beside the value.
    assert all(f"reading_{symbol}" in values["values"][0] for symbol in values["summary"]["mean_abs"])
    assert values["top_walks"] == []

    walked = subprocess.run(
        [sys.executable, "scripts/explain.py", "--run", str(first), "--method", "walks",
         "--scope", "engine", "--every", "60", "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert walked.returncode == 0, walked.stdout[-2000:] + walked.stderr[-2000:]
    stored = playground.run_detail(tmp_path, "first")["explanations"]
    assert sorted(x["method"] for x in stored) == ["shap", "walks"]
    name = next(x["name"] for x in stored if x["method"] == "walks")
    walks = playground.explanation_detail(tmp_path, "first", name)
    summary = walks["summary"]
    # Nothing a page drawing sensor values in cycles reads is written, so an
    # older page fails loudly instead of drawing score shares as cycles.
    assert not {"mean_abs", "sum_gap_max", "baseline", "baseline_prediction"} & set(summary)
    # Nothing shaped by the Shapley explanation either: no average-size ranking.
    assert "mean_abs_start" not in summary
    assert summary["value_unit"] == "score before the sigmoid"
    assert summary["conservation_error_max"] < 1e-9
    assert summary["gamma"] == [0.0, 1.5, 0.0]
    assert len(walks["values"]) == summary["windows"]
    row = walks["values"][0]
    symbols = summary["sensors"]
    assert not set(symbols) & set(row) and "baseline" not in row
    for window in walks["values"]:
        starts = sum(window[f"start_{s}"] for s in symbols)
        links = sum(v for k, v in window.items() if k.startswith("link_"))
        assert starts + window["constant"] == pytest.approx(window["score"], abs=1e-9)
        assert window["stays"] + window["moves"] == pytest.approx(starts, abs=1e-9)
        assert links == pytest.approx(summary["steps"] * starts, abs=1e-9)
    assert all(f"reading_{s}" in row for s in symbols)
    assert 1 <= len(walks["top_walks"]) <= 10
    assert (first / "explanations" / name / "top_walks.csv").is_file()
    # The GNN-LRP paper's own check is stored with every walk explanation.
    rankings = {"GNN-LRP, walks", "GNN-LRP, starting sensors", "GNN-GI, walks",
                "GNN-GI, starting sensors", "random"}
    assert {r["ranking"] for r in walks["flipping"]} == rankings
    assert len(walks["flipping"]) == summary["windows"] * 2 * len(rankings)
    assert {(a["task"], a["ranking"]) for a in summary["flipping_aufc"]} == {
        (task, ranking) for task in ("activation", "pruning") for ranking in rankings}


# -- the walk rule's setting in the Explain tab ------------------------------------


def _script(name: str):
    """One of the scripts in scripts/, imported as a module so its names can be read."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_walk_rule_settings_say_what_they_are_for_in_plain_words():
    """The setting and each of its options carry an i: what it is for in plain
    words first, never citing, then under a line where it comes from, and a
    Read more that the test above checks leads to a heading that exists."""
    import re

    cited = re.compile(r"\b[ADFP]-\d+\b|\bTable \d|\bEq\. ?\d|\bSection \d|\bFig\. ?\d")
    texts = [("gamma", playground.GAMMA_ABOUT)]
    texts += [(f"gamma={s['value']}", s["about"]) for s in playground.GAMMA_SETTINGS]
    for name, about in texts:
        assert about and about == about.strip(), name
        assert not cited.search(about), (name, cited.search(about).group(0))
        assert len(about.split()) <= 100, (name, len(about.split()))
    for setting in playground.GAMMA_SETTINGS:
        assert setting["source"]["kind"] in ("assumption", "extension"), setting["value"]
        assert setting["source"]["ref"] == "A-29", setting["value"]
    # The options come from different places, so, as for the Design tab's
    # temporal stage and graph, each carries a source and the row none; why
    # the first graph layer gets 0 is said once, in the row's words, not again
    # in every option's. Only the authors' option names that layer, so that
    # its label does not claim the authors' own number there.
    assert sum("can be negative" in about for _, about in texts) == 1
    assert "first graph layer" in playground.GAMMA_ABOUT and "can be negative" in playground.GAMMA_ABOUT
    assert [s["value"] for s in playground.GAMMA_SETTINGS if "first graph layer" in s["about"]] == ["authors"]
    offered = playground.options()
    assert offered["gamma_about"] == playground.GAMMA_ABOUT
    assert "gamma_source" not in offered
    assert offered["explain_more"]["gamma"] == playground.read_more("explain=gamma")
    assert all(s["more"] == offered["explain_more"]["gamma"] for s in offered["gamma_settings"])
    assert [s["value"] for s in offered["gamma_settings"]] == [
        s["value"] for s in playground.GAMMA_SETTINGS]


def test_the_page_offers_exactly_the_settings_the_script_takes():
    """One list on each side of the command line, the authors' setting first on both."""
    explain = _script("explain")
    assert tuple(s["value"] for s in playground.GAMMA_SETTINGS) == explain.GAMMA_SETTINGS
    assert explain.GAMMA_SETTINGS[0] == "authors"
    assert set(explain.FLIPPING_NAMES) == set(explain.GAMMA_SETTINGS)
    # The page draws the setting from the server's list, sends it for walks
    # only, and names a third setting's check by the label the form uses.
    script = (ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets" / "playground.js").read_text(
        encoding="utf-8")
    assert 'optionList("pg-e-gamma", S.options.gamma_settings' in script
    assert 'if (e.method === "walks") { request.gamma = explainGamma(); }' in script
    assert all(name.startswith("GNN-LRP ") for key, name in explain.FLIPPING_NAMES.items()
               if key not in ("authors", "gnn-gi"))
    assert "/^GNN-LRP (\\S+), (walks|starting sensors)$/" in script


def test_a_walk_request_carries_a_named_setting_and_never_numbers(tmp_path: Path):
    run = tmp_path / "reports" / "somerun"
    (run / "seed_42").mkdir(parents=True)
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "seed_42" / "model.pt").write_bytes(b"")
    base = {"run": "somerun", "seed": 42, "method": "walks", "scope": "engine", "engine": 7, "every": 1}

    default = playground.validate_explain(base, tmp_path)
    assert default["gamma"] == "authors"
    assert "--gamma" not in playground.commands.explain_arguments(default, None)
    assert playground.explanation_label(default) == "Walks through the sensor graph · engine 7 · seed 42"

    chosen = playground.validate_explain({**base, "gamma": "flat"}, tmp_path)
    command = playground.commands.explain_arguments(chosen, None, display=True)
    assert command[-2:] == ["--gamma", "flat"]
    assert playground.explanation_label(chosen) == (
        "Walks through the sensor graph · engine 7 · seed 42 · equal extra weight after the first layer")
    # Read back from a job's folder, the command names the same setting.
    read = playground.jobs._explain_request(tmp_path, " ".join(command), {})
    assert read["gamma"] == "flat"

    # Numbers, the authors' unmodified 3 at the first layer among them, are refused.
    for refused in ([3.0, 1.5, 0.0], [0.0, 1.0, 1.0], 3, "3", "0 1 1", "linspace"):
        with pytest.raises(playground.SettingsError, match="gamma must be one of"):
            playground.validate_explain({**base, "gamma": refused}, tmp_path)
    with pytest.raises(playground.SettingsError, match="walk explanations only"):
        playground.validate_explain({**base, "method": "shap", "gamma": "flat"}, tmp_path)
    assert playground.validate_explain({**base, "method": "shap"}, tmp_path)["gamma"] is None


def test_every_named_setting_gives_the_first_graph_layer_zero():
    """Whatever the name and however many graph layers, the first gets 0; for
    three layers the settings are the ones scripts/measure_walks.py compares."""
    pytest.importorskip("torch")
    explain = _script("explain")
    measured = _script("measure_walks").GAMMAS
    for setting in explain.GAMMA_SETTINGS:
        for layers in range(1, 7):
            values = explain.gamma_values(setting, layers)
            assert len(values) == layers
            assert values[0] == 0.0, (setting, layers, values)
            assert min(values) >= 0.0
    assert explain.gamma_values("authors", 3) == [0.0, 1.5, 0.0]
    assert explain.gamma_values("gnn-gi", 3) == list(measured["0"])
    assert explain.gamma_values("falling", 3) == list(measured["0_0.5_0.25"])
    assert explain.gamma_values("flat", 3) == list(measured["0_1_1"])
    with pytest.raises(ValueError, match="gamma must be one of"):
        explain.gamma_values("0 1 1", 3)


@pytest.fixture(scope="module")
def summary_run(tmp_path_factory, raw_dir):
    """A stored run of one untrained network whose temporal stage hands the
    graph three numbers per sensor, so a gamma at the first graph layer would
    change its walks; written the way train.py writes a run, never trained."""
    torch = pytest.importorskip("torch")
    import pandas as pd

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.models import ModelConfig, build_model

    run = tmp_path_factory.mktemp("reports") / "fd001_physics_row_cap_1seed_varsplit_window_summary_test"
    run.mkdir()
    settings = PreprocessingConfig(window_size=20, validation_fraction=0.25, seed=42)
    settings.save(run / "preprocessing_config.yaml")
    config = ModelConfig(temporal="window_summary")
    config.save(run / "model_config.yaml")
    pd.DataFrame({"seed": [42], "rmse": [20.0]}).to_csv(run / "results.csv", index=False)
    (run / "summary.json").write_text(json.dumps({
        "seeds": 1, "split": "varies_with_seed", "temporal_stage": "window_summary",
        "finished_utc": "20261002T000000Z"}), encoding="utf-8")
    prepared = run_preprocessing(settings, data_dir=raw_dir)
    torch.manual_seed(42)
    model = build_model(prepared.adjacency_normalized, config, window_size=20, rul_max=125)
    (run / "seed_42").mkdir()
    torch.save(model.state_dict(), run / "seed_42" / "model.pt")
    return run


@pytest.mark.parametrize("setting", ["authors", "gnn-gi", "falling", "flat"])
def test_no_positive_gamma_reaches_the_first_graph_layer(summary_run, raw_dir, monkeypatch, setting):
    """Every gamma the rule is ever run with, recorded where it is applied, is 0
    at the first graph layer; the stored explanation records the setting."""
    pytest.importorskip("torch")
    from thesis_gnn_rul_xai.explain import walks

    explain = _script("explain")
    monkeypatch.setattr(explain, "capture_code_state", lambda: {"status": [], "patch": "", "untracked": {}})
    monkeypatch.setattr(explain, "current_commit", lambda: "test")
    applied: list[list[float]] = []
    original = walks._rule_forward

    def recording(model, nodes, gammas):
        applied.append([float(g) for g in gammas])
        return original(model, nodes, gammas)

    monkeypatch.setattr(walks, "_rule_forward", recording)
    arguments = ["explain.py", "--run", str(summary_run), "--method", "walks", "--scope", "engine",
                 "--every", "100", "--device", "cpu", "--data-dir", str(raw_dir), "--quiet"]
    if setting != "authors":
        arguments += ["--gamma", setting]
    monkeypatch.setattr(sys, "argv", arguments)
    assert explain.main() == 0

    chosen = explain.gamma_values(setting, 3)
    assert applied and all(gammas[0] == 0.0 for gammas in applied), applied
    assert chosen in applied

    folder = max((summary_run / "explanations").glob("walks_*"), key=lambda p: p.stat().st_mtime)
    assert ("_gamma-" + setting + "_" in folder.name) == (setting != "authors")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["gamma_setting"] == setting
    assert summary["gamma"] == chosen
    assert summary["conservation_error_max"] < 1e-9
    rankings = {r["ranking"] for r in summary["flipping_aufc"]}
    expected = {"GNN-LRP", "GNN-GI"} | ({explain.FLIPPING_NAMES[setting]} if setting in ("falling", "flat") else set())
    assert rankings == {f"{name}, {what}" for name in expected for what in ("walks", "starting sensors")} | {"random"}


def test_a_gamma_at_the_first_layer_would_change_the_walks_of_that_network(summary_run, raw_dir):
    """The guard above has teeth: on this network the authors' unmodified 3 at
    the first graph layer, which walk_relevance refuses, moves the walks."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from thesis_gnn_rul_xai.explain import walk_relevance
    from thesis_gnn_rul_xai.explain.walks import _chunk
    from thesis_gnn_rul_xai.training import load_seed, read_run

    model, prepared = load_seed(read_run(summary_run), 42, data_dir=raw_dir)
    windows = prepared.test.features[:4]
    with torch.no_grad():
        guarded = _chunk(model, torch.from_numpy(windows.astype(np.float32)), [0.0, 1.0, 1.0])[0].numpy()
        unguarded = _chunk(model, torch.from_numpy(windows.astype(np.float32)), [3.0, 1.0, 1.0])[0].numpy()
    assert not np.allclose(guarded, unguarded)
    with pytest.raises(ValueError, match="must be 0 at the first graph layer"):
        walk_relevance(model, windows, gamma=[3.0, 1.0, 1.0])


def test_the_explain_form_offers_the_rule_for_walks_only_and_names_the_choice_once():
    """Run in Node: the setting's row is hidden unless walks are chosen, the
    request carries its name only for walks, and a stored explanation, its
    table row and its check name the setting in the form's own words."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to run the page's script.")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const options = JSON.parse(fs.readFileSync(0, 'utf8'));
const sent = [];
const context = {window: {}, fetch(url, init) { sent.push([url, JSON.parse(init.body)]); return new Promise(() => {}); }};
vm.createContext(context);
const source = fs.readFileSync('./src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {S, explainForm, submitExplanation, gammaNote, methodLabel, flipName,
    setup: function (helpers) { T = helpers; }};`), context);
const P = context.window.test;
const escape = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
P.setup({escapeHtml: escape, fixed: (x, places) => Number(x).toFixed(places), toast() {}});
P.S.options = options; P.S.infos = 0;
const run = {name: 'r', finished: '20261002T000000Z', subset: 'FD001', temporal: 'last_cycle', graph: 'physics', seeds: 5};
P.S.explain = {run: 'r', seed: '42', method: 'shap', scope: 'engine', engine: '', every: 1, gamma: null};
let html = P.explainForm(run, ['42'], [run]);
assert.ok(html.includes('data-row="pg-e-gamma" hidden'), 'the rule is hidden for Shapley values');
P.submitExplanation();
assert.equal(sent.at(-1)[1].gamma, undefined);
P.S.explain.method = 'walks';
html = P.explainForm(run, ['42'], [run]);
assert.ok(html.includes('data-row="pg-e-gamma">'), 'the rule is shown for walks');
for (const s of options.gamma_settings) {
  assert.ok(html.includes('name="pg-e-gamma" value="' + s.value + '"' + (s.value === 'authors' ? ' checked' : '')), s.value);
  assert.equal(html.split(escape(s.about)).length - 1, 1, 'each option says what it does once: ' + s.value);
}
assert.equal(html.split(escape(options.gamma_about)).length - 1, 1);
for (const s of options.gamma_settings) {
  assert.equal(html.split(escape(s.source.text)).length - 1, 1, 'each option names its source once: ' + s.value);
}
P.submitExplanation();
assert.equal(sent.at(-1)[1].gamma, 'authors');
P.S.explain.gamma = 'flat'; P.submitExplanation();
assert.deepEqual([sent.at(-1)[0], sent.at(-1)[1].gamma], ['/api/playground/explain', 'flat']);

assert.equal(P.methodLabel({method: 'walks', gamma_setting: 'falling', gamma: [0, 0.5, 0.25]}),
  'Walks through the sensor graph · halving extra weight');
assert.equal(P.methodLabel({method: 'walks', gamma_setting: 'authors', gamma: [0, 1.5, 0]}), 'Walks through the sensor graph');
assert.equal(P.methodLabel({method: 'walks', gamma: [0, 1.5, 0]}), 'Walks through the sensor graph');
assert.equal(P.methodLabel({method: 'walks', gamma: [0, 0, 0]}), 'Walks through the sensor graph · no extra weight');
assert.equal(P.methodLabel({method: 'shap'}), 'Exact Shapley values');
// The option's name is in the explanation's table row; the note gives its numbers.
const note = P.gammaNote({gamma_setting: 'flat', gamma: [0, 1, 1]});
// A setting is recorded whether the form or the command line chose it, so
// the note never says which of the two did.
assert.ok(note.includes('0.00, 1.00, 1.00') && note.includes('chosen when this explanation was made'), note);
assert.ok(!note.toLowerCase().includes('equal extra weight'), note);
assert.ok(P.gammaNote({gamma: [0, 0, 0]}).includes('Explain the run again'));
const baseline = P.gammaNote({gamma_setting: 'gnn-gi', gamma: [0, 0, 0]});
assert.ok(baseline.includes('chosen when it was made'), baseline);
for (const text of [note, baseline]) { assert.ok(!text.includes('Explain form'), text); }
assert.equal(P.flipName('GNN-LRP flat, walks'), 'Whole walks, equal extra weight after the first layer');
assert.equal(P.flipName('GNN-LRP falling, starting sensors'), 'Sensor totals, halving extra weight');
assert.notEqual(P.flipName('GNN-LRP falling, starting sensors'), P.flipName('GNN-GI, starting sensors'));
assert.equal(P.flipName('GNN-LRP, walks'), "Whole walks, the authors' setting");
assert.equal(P.flipName('GNN-LRP, starting sensors'), "Sensor totals, the authors' setting");
console.log('The rule setting passed.');
"""
    result = subprocess.run([node, "-e", script], cwd=ROOT, input=json.dumps(playground.options()),
                            text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_a_chosen_setting_s_check_is_drawn_in_a_colour_far_from_the_baseline():
    """The flipping curves of a third setting are gold: not orange, which the
    walk view already uses for a negative share, and not the accent blue,
    which sits close to the baseline's blue-green. The distances the
    playground README quotes are the straight-line distance in CIE Lab
    between the colours app.css defines, light mode first, and the contrasts
    are against the figure's own background."""
    import math
    import re

    assets = ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets"
    css = (assets / "app.css").read_text(encoding="utf-8")

    def colours(name: str) -> list[str]:
        return re.findall(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", css)

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def channels(hex_colour: str) -> tuple[float, float, float]:
        return tuple(linear(int(hex_colour[i:i + 2], 16) / 255) for i in (1, 3, 5))

    def lab(hex_colour: str) -> tuple[float, float, float]:
        r, g, b = channels(hex_colour)
        x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

        def f(t: float) -> float:
            return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

        return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))

    def contrast(one: str, other: str) -> float:
        lighter, darker = sorted((0.2126 * r + 0.7152 * g + 0.0722 * b
                                  for r, g, b in (channels(one), channels(other))), reverse=True)
        return (lighter + 0.05) / (darker + 0.05)

    def distances(name: str, to: str) -> list[int]:
        return [round(math.dist(lab(a), lab(b))) for a, b in zip(colours(name), colours(to), strict=True)]

    named = ("fig-cool", "fig-accent", "fig-core", "fig-ink", "page")
    assert all(len(colours(name)) == 2 for name in named), {name: colours(name) for name in named}
    assert distances("fig-accent", "fig-cool") == [32, 23]
    assert distances("fig-core", "fig-cool") == [78, 83]
    assert distances("fig-core", "fig-ink") == [69, 63]
    # A drawn line needs 3 to 1 against what it is drawn on: a figure's .pg-figure box has --page.
    ratios = [round(contrast(line, ground), 1) for line, ground in zip(colours("fig-core"), colours("page"),
                                                                       strict=True)]
    assert ratios == [4.9, 9.2] and min(ratios) >= 3
    assert ".pg-figure { margin: 0;" in css and "background: var(--page)" in css.split(".pg-figure { margin: 0;")[1].split("}")[0]

    script = (assets / "playground.js").read_text(encoding="utf-8")
    assert '{ colour: "var(--fig-core)", width: 2.2 } : { colour: "var(--fig-core)", dash: "6 4" }' in script
    # Orange keeps its one meaning in this view: a negative share.
    assert "orange, negative numbers subtract from it" in script
    flip_style = script.split("function flipStyle(ranking) {")[1].split("\n  }\n")[0]
    assert "fig-warn" not in flip_style and "fig-accent" not in flip_style


# --- the network table: layer_summary() as it comes (2026-10-03) ----------


@pytest.mark.parametrize(
    ("temporal", "source", "learned"), [("lstm", "LSTM", 62_753), ("gru", "GRU", 47_073), ("last_cycle", "", 0)]
)
def test_the_network_table_is_layer_summary_with_the_stage_in_the_input_row(temporal, source, learned):
    """The page draws the rows the network gives, so the stage's count and name must already be on Input."""
    pytest.importorskip("torch")
    answer = playground.preview(playground.validate_settings({"temporal": temporal}))
    layers = answer["layers"]
    assert [row["layer"] for row in layers] == ["Input", "GCN-1", "GCN-2", "GCN-3", "Flatten", "FC-1", "FC-2"]
    assert (layers[0]["from"], layers[0]["out"], layers[0]["parameters"]) == (source, "(B, 14, 1)", learned)
    assert sum(int(row["parameters"]) for row in layers) == answer["parameters"]
    # The page joins no rows of its own any more; it shows each row's name, its "from", its shape and count.
    script = (ROOT / "src/thesis_gnn_rul_xai/docsite/assets/playground.js").read_text(encoding="utf-8")
    assert "table4Rows" not in script and '"Temporal"' not in script
    assert "preview.layers.map(function (l) {" in script and "esc(l.from)" in script


# --- the page calls a capped label capped (added 2026-10-03) ------------------

CAPPED_RUN = "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z"


def test_the_page_calls_a_capped_test_label_capped_not_true(raw_dir):
    """Seed 42 of the stored default run, named by the page's own code.

    The run capped its test labels at 125. Test engine 1 has 112 cycles left,
    under the cap, so its label is 112 and the slider reads "engine 1, capped
    remaining life 112, predicted 108.6". Eleven of the 100 labels read 125
    where NASA's answer file gives 126 to 145, so the page may not call that
    column the true remaining life; it says so only for a run that left its
    test labels uncapped, and an engine's own windows are always capped.
    """
    import shutil

    import numpy as np

    from thesis_gnn_rul_xai.data.loading import read_rul_file

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to run the page's naming code.")
    if not (ROOT / "reports" / CAPPED_RUN / "seed_42" / "test_predictions.csv").is_file():
        pytest.skip(f"The stored run {CAPPED_RUN} is not in reports/.")
    detail = playground.run_detail(ROOT, CAPPED_RUN)
    rows = detail["per_seed"]["42"]["predictions"]
    answers = read_rul_file(raw_dir / "RUL_FD001.txt")
    labels = np.array([row["rul"] for row in rows])
    truth = answers.loc[[row["unit"] for row in rows]].to_numpy()
    np.testing.assert_array_equal(labels, np.minimum(truth, 125))
    assert sorted(truth[labels < truth].tolist()) == [126, 128, 128, 131, 135, 136, 137, 137, 137,
                                                      142, 145]

    script = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {window: {}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {S, lifeLabel, lifeSeriesName, explanationCapped, explanationCaption, predictionsChartName,
    setup: function (helpers) { T = helpers; }};`), context);
const P = context.window.test;
P.setup({escapeHtml: s => String(s), fixed: (x, n) => Number(x).toFixed(n)});
const d = input.detail, row = input.row;
P.S.explain = P.S.explain || {};
P.S.explain.detail = d;
P.S.explain.values = {summary: {scope: 'test'}};
const out = {series: P.lifeSeriesName(d.clip_test_labels, d.rul_max), order: P.lifeLabel(d.clip_test_labels),
             caption: P.explanationCaption(row),
             engine: P.explanationCapped({summary: {scope: 'engine'}})};
P.S.explain.detail = Object.assign({}, d, {clip_test_labels: false});
out.uncapped = P.explanationCaption(row);
P.S.explain.detail = Object.assign({}, d, {clip_test_labels: undefined});
out.unrecorded = P.explanationCaption(row);
out.engineWhenUncapped = P.explanationCapped({summary: {scope: 'engine'}});
out.chart = P.predictionsChartName(d.clip_test_labels, d.rul_max);
out.chartUncapped = P.predictionsChartName(false, d.rul_max);
out.chartUnrecorded = P.predictionsChartName(undefined, d.rul_max);
process.stdout.write(JSON.stringify(out));
"""
    payload = {"detail": {key: detail[key] for key in ("clip_test_labels", "rul_max")}, "row": rows[0]}
    result = subprocess.run([node, "-e", script], cwd=ROOT, input=json.dumps(payload),
                            text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr
    named = json.loads(result.stdout)
    assert (rows[0]["unit"], rows[0]["rul"], round(rows[0]["predicted"], 1)) == (1, 112, 108.6)
    assert named == {
        "series": "remaining life capped at 125",
        "order": "capped remaining life",
        "caption": "engine 1, capped remaining life 112, predicted 108.6",
        "engine": True,
        "uncapped": "engine 1, true remaining life 112, predicted 108.6",
        "unrecorded": "engine 1, test label 112, predicted 108.6",
        "engineWhenUncapped": True,
        # The Fig. 8 chart's spoken name (added 2026-10-03), after the series beside it.
        "chart": "predictions against the remaining life capped at 125",
        "chartUncapped": "predictions against the true remaining life",
        "chartUnrecorded": "predictions against the test label",
    }
    source = (ROOT / "src/thesis_gnn_rul_xai/docsite/assets/playground.js").read_text(encoding="utf-8")
    assert "against truth" not in source
    assert "label: predictionsChartName(d.clip_test_labels, d.rul_max)" in source


# --- comparing two runs when one seed has no ASUE (assumption A-39) ---


def test_a_run_comparison_keeps_each_seed_with_its_own_difference_and_leaves_an_undefined_mean_undefined(tmp_path):
    """With seed 43's ASUE not a number, seeds 44 to 46 keep their own differences and the mean is None."""
    import shutil

    import numpy as np
    import pandas as pd

    first = "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z"
    second = "fd001_physics_row_cap_5seed_varsplit_noearlystop_cfgba82e3_f4f40ad_20261002T170226Z"
    if not all((ROOT / "reports" / name / "results.csv").is_file() for name in (first, second)):
        pytest.skip("the two stored runs this test names are not in reports/")
    # Copies of the two runs' records, never the runs themselves (working rule 3a).
    for name in (first, second):
        (tmp_path / "reports" / name).mkdir(parents=True)
        for record in ("summary.json", "results.csv", "preprocessing_config.yaml", "model_config.yaml"):
            shutil.copy(ROOT / "reports" / name / record, tmp_path / "reports" / name / record)

    # Both runs as stored: five seeds paired, the default against the same
    # network trained all 200 passes; second minus first, seed by seed.
    whole = playground.compare_runs(tmp_path, first, second)["measures"]["asue"]
    assert [(row["seed"], round(row["difference"], 4)) for row in whole["per_seed"]] == [
        (42, -0.3967), (43, 1.5287), (44, -1.3184), (45, 0.4724), (46, 0.0)]
    assert round(whole["mean_difference"], 4) == 0.0572   # (-0.3967 + 1.5287 - 1.3184 + 0.4724 + 0) / 5

    # Seed 43 of the second run with no engine predicted early: its ASUE is
    # not a number, which results.csv holds as an empty field.
    results = pd.read_csv(tmp_path / "reports" / second / "results.csv")
    results.loc[results["seed"] == 43, "asue"] = np.nan
    results.to_csv(tmp_path / "reports" / second / "results.csv", index=False)
    asue = playground.compare_runs(tmp_path, first, second)["measures"]["asue"]
    assert [(row["seed"], None if row["difference"] is None else round(row["difference"], 4))
            for row in asue["per_seed"]] == [(42, -0.3967), (43, None), (44, -1.3184), (45, 0.4724), (46, 0.0)]
    # Undefined, as a mean over seeds is when a seed has no ASUE, rather than
    # the -0.3107 of the four seeds that have one.
    assert (asue["mean_difference"], asue["t"], asue["p"]) == (None, None, None)


def test_one_random_graph_for_every_seed_with_nothing_held_back_is_refused_before_a_job_starts(monkeypatch):
    """scripts/train.py refuses --fixed-split with a random or rewired graph and no engine held back; so does the form."""
    conflict = {"fixed_split": True, "validation_fraction": 0.0, "early_stopping": False}
    # Today the form offers neither graph, so both are refused by their name first.
    for graph in ("random", "rewired"):
        with pytest.raises(playground.SettingsError, match="Sensor graph must be one of"):
            playground.validate_settings({**conflict, "graph": graph})

    # Were it to offer them, the combination would still be refused, in the
    # training script's terms, and only the combination.
    from thesis_gnn_rul_xai.docsite.playground.settings import _BY_KEY

    offered = _BY_KEY["graph"]["choices"]
    monkeypatch.setitem(_BY_KEY["graph"], "choices", [
        *offered, {"value": "random", "label": "random", "source": None},
        {"value": "rewired", "label": "rewired", "source": None}])
    for graph in ("random", "rewired"):
        with pytest.raises(playground.SettingsError, match="needs engines held back"):
            playground.validate_settings({**conflict, "graph": graph})
        held_back = playground.validate_settings({**conflict, "graph": graph, "validation_fraction": 0.2})
        assert [note["text"] for note in playground.caveats(held_back) if note["ref"] == "A-41"] == [
            "The graph is drawn at random from one seed and serves every seed."]
        playground.validate_settings({**conflict, "graph": graph, "fixed_split": False})
    for graph in ("physics", "complete", "correlation"):
        playground.validate_settings({**conflict, "graph": graph})


def test_every_stored_run_has_an_address_that_leads_back_to_it():
    """A run's address names its folder, a nested folder keeps its slashes, and reading it back gives the same run."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the playground's addresses.")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const context = {window: {}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {runAddress, readAddress};`), context);
const P = context.window.test;
const flat = 'fd001_physics_row_cap_5seed_varsplit_net9f3068_cfgba82e3_19dceb5_20261002T225228Z';
assert.equal(P.runAddress(flat), '#/explore/playground/runs/' + flat);
const nested = 'sensitivity_study/runs/FD002/reference';
assert.equal(P.runAddress(nested), '#/explore/playground/runs/sensitivity_study/runs/FD002/reference');
for (const name of [flat, nested, 'a run#with?odd chars']) {
  const sub = P.runAddress(name).slice('#/explore/playground/'.length);
  const read = P.readAddress(sub);
  assert.equal(read.tab, 'results');
  assert.equal(read.selected, name);
}
assert.deepEqual(JSON.parse(JSON.stringify(P.readAddress('runs'))), {tab: 'results', selected: null});
assert.deepEqual(JSON.parse(JSON.stringify(P.readAddress('explain'))), {tab: 'explain', selected: null});
assert.deepEqual(JSON.parse(JSON.stringify(P.readAddress(''))), {tab: 'design', selected: null});
"""
    rendered = subprocess.run([node, "-e", script], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    assert rendered.returncode == 0, rendered.stderr


def test_a_run_shows_one_part_at_a_time_and_lists_its_explanations():
    """The address names the part, the notes stay with the results, and a stored explanation's values stay off the run's page."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the parts of a run's page.")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const context = {window: {location: {hash: ''}}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {runTabInAddress, runTabs, runOverview, runExplanations, runNotesPointer,
    setup: function (helpers, state) { T = helpers; S = state; }};`), context);
const P = context.window.test;
const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const fixed = (value, places) => value === null || value === undefined || isNaN(value)
  ? '—' : Number(value).toFixed(places === undefined ? 2 : places);
P.setup({escapeHtml, fixed}, {options: {gamma_settings: [{value: 'authors', label: 'The authors setting'}]}, explain: {}});

// The part of the run an address names; an address from before the parts names a section.
const run = '#/explore/playground/runs/fd001_run';
const part = hash => { context.window.location.hash = hash; return P.runTabInAddress(); };
assert.equal(part(run), 'overview');
assert.equal(part(run + '#charts'), 'charts');
assert.equal(part(run + '#explanations'), 'explanations');
assert.equal(part(run + '#settings'), 'settings');
assert.equal(part(run + '#pg-run-settings'), 'settings');
assert.equal(part(run + '#pg-run-interpretation'), 'overview');
assert.equal(part(run + '#no-such-part'), 'overview');

const stored = [
  {name: 'walks_test_seed42', method: 'walks', scope: 'test', source: 'test', seed: 42, windows: 100,
   finished_utc: '20261102T121443Z', commit: 'a7c23d2', gamma_setting: 'authors'},
  {name: 'shap_engine_unit69_seed42', method: 'shap', scope: 'engine', source: 'training', unit: 69, every: 1, seed: 42,
   windows: 333, finished_utc: '20261102T110025Z', commit: 'a7c23d2', baseline_prediction: 88.688,
   mean_abs: {P30: 14.583, T50: 9.567}}
];
const record = {name: 'fd001_run', has_weights: true, explanations: stored, per_seed: {'42': {}, '43': {}},
  mean: {rmse: 16.737, score: 522.0, r2: 0.826, asue: 12.471}, std: {rmse: 0.435, score: 95.0, r2: 0.009, asue: 0.818},
  paper_row: {label: 'ESTGNN w/o LSTM', source: 'Table 11', rmse: 14.93, score: null, r2: null, asue: null}};

// The tab carries how many explanations are stored.
const tabs = P.runTabs(record, 'explanations');
assert.match(tabs, /id="pg-run-tab-explanations"[^>]*aria-selected="true"/);
assert.match(tabs, /2 explanations stored for this run/);

// Each stored explanation is a row that opens it; none of its values is repeated here.
const list = P.runExplanations(record);
assert.equal(list.split('data-action="open-explanation"').length - 1, 2);
assert.ok(list.includes('data-run="fd001_run"') && list.includes('data-name="shap_engine_unit69_seed42"'));
assert.ok(list.includes('Exact Shapley values') && list.includes('Training engine 69'));
assert.ok(list.includes('Walks through the sensor graph') && list.includes('Test engines'));
assert.ok(list.includes('2026-11-02 12:14 UTC') && list.includes('a7c23d2'));
assert.ok(!list.includes('14.583') && !list.includes('88.688'));
assert.ok(P.runExplanations({name: 'r', has_weights: true, explanations: []}).includes('None yet'));
assert.ok(P.runExplanations({name: 'r', has_weights: false, explanations: []}).includes('none can be made'));

// The notes stand on the same part as the results they qualify, in full.
const cautions = [{ref: 'F-16', text: 'Saved choice <b>', level: 'warn'}, {ref: 'A-02', text: 'Capped answers', level: 'info'}];
const overview = P.runOverview(record, cautions);
assert.ok(overview.indexOf('class="pg-metrics"') < overview.indexOf('How to interpret this run'));
assert.ok(overview.includes('Saved choice &lt;b&gt;') && overview.includes('Capped answers'));
assert.ok(overview.includes('#/entry/F-16') && overview.includes('Choosing the reported weights'));
assert.ok(overview.includes('16.737') && overview.includes('± 0.435') && overview.includes('14.93') && overview.includes('+1.81'));
assert.ok(overview.includes('Mean ± standard deviation over 2 seeds.'));
assert.ok(overview.includes('Compared with ESTGNN w/o LSTM, Table 11, which prints RMSE only.'));
assert.ok(overview.includes('Lower is better for RMSE, Score and ASUE; higher is better for R².'));
// "The paper prints none" is said once below the tiles when the paper prints RMSE only,
// and in each tile that lacks a value when it prints more than RMSE.
assert.ok(!overview.includes('The paper prints none'));
const fuller = P.runOverview(Object.assign({}, record, {paper_row: {label: 'ESTGNN', source: 'Table 7', rmse: 11.78, score: 278.6,
  r2: null, asue: null}}), []);
assert.equal(fuller.split('The paper prints none').length - 1, 2);
assert.ok(!fuller.includes('How to interpret this run'));

// Where results are shown without the notes, one line counts them and leads to them.
const pointer = P.runNotesPointer(record, cautions);
assert.ok(pointer.includes('1 caution and 1 note apply to these results.'));
assert.ok(pointer.includes('data-part="overview"') && pointer.includes('data-reach="pg-run-interpretation"'));
assert.equal(P.runNotesPointer(record, []), '');
"""
    rendered = subprocess.run([node, "-e", script], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    assert rendered.returncode == 0, rendered.stderr


def test_an_explanation_opens_on_its_charts_with_one_control_for_each_thing():
    """What the author had removed stays removed, and the button that starts an explanation ends the form."""
    assets = ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets"
    script = (assets / "playground.js").read_text(encoding="utf-8")
    engine_view = (assets / "engine-view.js").read_text(encoding="utf-8")
    css = (assets / "app.css").read_text(encoding="utf-8")

    # The engine panel: the two layouts, and no control that hides the graph or clears a choice.
    for gone in ("Show graph connections", "Clear selection", "data-ev-overlay", "data-ev-clear"):
        assert gone not in engine_view, gone
    assert 'event.key === "Escape" && state.sensor' in engine_view

    # The view of a Shapley or occlusion explanation: no paragraph and no notes above the charts,
    # and no tab that repeats the engine panel's graph.
    view = script.split("function drawExplanation", 1)[1].split("function engineFigure", 1)[0]
    for gone in ("Equation 14 says", "Table 12 cannot hold", "not evidence that the sensor causes",
                 "share of the prediction's difference", "caveatList(", 'label: "Sensor graph"'):
        assert gone not in view, gone
    assert "function sensorGraph(" not in script
    assert view.count("Occlusion is a comparison added by this project; the paper does not use it.") == 1

    # The button is the form's last row, not a cell of the settings' grid.
    form = script.split("function explainForm", 1)[1].split("function explainNote", 1)[0]
    assert "cell({ control: '<button" not in form
    assert form.index('class="pg-form-actions"') < form.index('data-action="explain">Explain</button>')
    assert ".pg-form-actions {" in css

    # The graph layers, like the sub-dataset, have no Reset of their own.
    assert 'if (f.key === "subset" || f.key === "spatial") { return ""; }' in script


def test_the_values_over_time_are_drawn_per_sensor_and_in_three_dimensions():
    """One chart per sensor with numbered axes, the same values as ribbons in one space, and a name's meaning beside it."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the charts of an explanation.")
    stored = next((p for p in sorted((ROOT / "reports").glob("*/explanations/*/summary.json"))
                   if (s := json.loads(p.read_text(encoding="utf-8"))).get("method") == "shap"
                   and s.get("scope") == "engine" and s.get("windows", 0) >= 30), None)
    if stored is None:
        pytest.skip("A stored Shapley explanation of one engine of the real FD001 data is needed.")
    data = playground.explanation_detail(ROOT, stored.parents[2].name, stored.parent.name)
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const E = require('./src/thesis_gnn_rul_xai/docsite/assets/engine-view.js').ThesisEngineView;
const context = {window: {ThesisEngineView: E}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {sensorHistory, historyScale, historyColumns, ribbons, ribbonCamera, ribbonSize, beeswarm, signedValue,
    RIBBON_VIEW, setup: function (helpers, state) { T = helpers; S = state; }};`), context);
const P = context.window.test;
const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
P.setup({escapeHtml, fixed: (value, places) => Number(value).toFixed(places === undefined ? 2 : places)}, {explain: {}});
const rows = data.values, sensors = Object.keys(data.summary.mean_abs), n = sensors.length;
const count = (text, piece) => text.split(piece).length - 1;
assert.equal(n, 14);

// Four charts side by side on a wide screen, two on a tablet, one on a phone.
assert.deepEqual([1794, 1100, 760, 330].map(P.historyColumns), [4, 3, 2, 1]);

// One chart per sensor, each with numbers on both axes, the sensor's name and its value at the marked window.
const width = 1794, at = 3, scale = P.historyScale(rows, sensors, width);
const charts = P.sensorHistory(rows, sensors, at, width);
assert.equal(count(charts, '<figure class="pg-sm"'), n);
assert.ok(charts.includes('grid-template-columns:repeat(4,'));
assert.ok(count(charts, 'class="pg-tick"') >= n * 6);
for (const s of sensors) {
  assert.ok(charts.includes('<b data-sensor-tip="' + s + '">' + s + '</b><span class="pg-sm-value">' + P.signedValue(rows[at][s]) + '</span>'), s);
}
// A stored value stands where the two scales put it: the first sensor at the marked window.
const first = sensors[0], row = rows[at];
assert.ok(charts.includes(scale.X(row.end_cycle).toFixed(1) + ',' + scale.Y(row[first]).toFixed(1)));
// Every chart shares one vertical scale, wide enough for the largest value of any sensor.
const all = rows.flatMap(r => sensors.map(s => r[s]));
assert.ok(scale.lo < Math.min(...all) && scale.hi > Math.max(...all));
assert.ok(Math.abs(scale.largest - Math.max(...all.map(Math.abs))) < 1e-12);
assert.ok(Math.abs(scale.cycleAt(scale.X(row.end_cycle)) - row.end_cycle) < 1e-9);

// The same values in one space: a ribbon and a dot per sensor, and the three axes named.
const view = {az: P.RIBBON_VIEW.az, el: P.RIBBON_VIEW.el};
const chart = P.ribbons(rows, sensors, width, view, at, 'SHAP value');
assert.equal(count(chart, '<g class="pg-3d-ribbon"'), n);
assert.equal(count(chart, '<circle'), n);
for (const title of ['>Features</text>', '>SHAP value</text>', '>Cycle</text>']) assert.ok(chart.includes(title), title);
for (const s of sensors) assert.ok(chart.includes('class="pg-3d-name" data-sensor-tip="' + s + '">' + s + '</text>'), s);
assert.ok(!chart.includes('NaN'));
// Turning it draws it from another side, and leaves the number of ribbons alone.
const turned = P.ribbons(rows, sensors, width, {az: view.az + 90, el: 50}, at, 'SHAP value');
assert.notEqual(turned, chart);
assert.equal(count(turned, '<g class="pg-3d-ribbon"'), n);
// Nothing shrinks with distance: the same height is the same length on the screen at the front and at the back.
const size = P.ribbonSize(width), cam = P.ribbonCamera(view, size.W, size.H);
const rise = (u, v) => cam.at(u, v, 0)[1] - cam.at(u, v, 0.4)[1];
assert.ok(Math.abs(rise(-1, -1) - rise(1, 1)) < 1e-9 && rise(0, 0) > 0);

// Every value as one dot: a row is 38 units tall on a wide screen, and each name says what it stands for.
const swarm = P.beeswarm(rows, sensors, 1200);
assert.ok(swarm.includes('viewBox="0 0 1200 ' + (10 + 38 * n + 36) + '"'));
assert.equal(count(swarm, '<circle'), rows.length * n);
for (const s of sensors) assert.ok(swarm.includes('data-sensor-tip="' + s + '">' + s + '</text>'), s);
assert.ok(E.locations.T24.name.startsWith('Temperature'));
"""
    done = subprocess.run([node, "-e", script], input=json.dumps(data), text=True, encoding="utf-8",
                          capture_output=True, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr


def test_a_walk_explanation_s_tabs_draw_the_stored_numbers_and_say_what_they_are():
    """The score as a sum whose bars meet, the largest walks one position per layer, and the check as two tests."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the tabs of a walk explanation.")
    stored = next((p for p in sorted((ROOT / "reports").glob("*/explanations/*/summary.json"))
                   if (s := json.loads(p.read_text(encoding="utf-8"))).get("method") == "walks"
                   and s.get("scope") == "engine" and s.get("flipping_aufc") and s.get("sensors")), None)
    if stored is None:
        pytest.skip("A stored walk explanation of one engine, with its check, is needed.")
    data = playground.explanation_detail(ROOT, stored.parents[2].name, stored.parent.name)
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const E = require('./src/thesis_gnn_rul_xai/docsite/assets/engine-view.js').ThesisEngineView;
const context = {window: {ThesisEngineView: E}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {scoreSum, scoreSumScale, walkParts, walkPartList, topWalks, flippingFigure, ownRanking, scoreShare,
    setup: function (helpers, state) { T = helpers; S = state; }};`), context);
const P = context.window.test;
const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
P.setup({escapeHtml, fixed: (value, places) => Number(value).toFixed(places === undefined ? 2 : places)}, {explain: {}, options: {}});
const rows = data.values, sum = data.summary, count = (text, piece) => text.split(piece).length - 1;
const at = Math.min(7, rows.length - 1), row = rows[at];

// The score, added up: three named parts and their total, each with the stored number.
const parts = P.walkPartList(sum);
assert.equal(parts.map(p => p.key).join('|'), 'constant|stays|moves');
const home = (sum.groups || []).length ? 'group' : 'sensor';
assert.equal(parts.map(p => p.name).join('|'), 'Added by the network|Walks that stay in their ' + home + '|Walks that cross to another ' + home);
const added = P.scoreSum(data, at);
assert.ok(added.includes('The score of cycle ' + row.end_cycle + ' of engine ' + row.unit + ', added up'));
const shown = [...added.matchAll(/class="pg-sum-value">([^<]+)</g)].map(m => m[1]);
assert.deepEqual(shown, [row.constant, row.stays, row.moves, row.score].map(P.scoreShare));
// Each bar starts where the one above ended, and the total's bar ends where the last part's does.
const bars = [...added.matchAll(/class="pg-sum-bar[^"]*" style="left:([\d.]+)%;width:([\d.]+)%/g)].map(m => [Number(m[1]), Number(m[1]) + Number(m[2])]);
assert.equal(bars.length, 4);
const scale = P.scoreSumScale(rows), place = v => (v - scale.lo) / (scale.hi - scale.lo) * 100;
let total = 0;
[row.constant, row.stays, row.moves].forEach((value, i) => {
  const from = total; total += value;
  assert.ok(Math.abs(bars[i][0] - place(Math.min(from, total))) < 0.02, 'part ' + i + ' starts at the running total');
  assert.ok(Math.abs(bars[i][1] - Math.max(place(Math.max(from, total)), bars[i][0] + 0.2)) < 0.03, 'part ' + i + ' ends at the next one');
});
assert.ok(Math.abs(total - row.score) < 1e-9, 'the stored parts add up to the stored score');
assert.ok(Math.abs(bars[3][0] - place(Math.min(0, row.score))) < 0.02 && Math.abs(bars[3][1] - place(Math.max(0, row.score))) < 0.03);
// One scale holds every running total of every cycle, so bars compare as the slider moves.
for (const r of rows) {
  for (const v of [r.constant, r.constant + r.stays, r.score]) assert.ok(v > scale.lo && v < scale.hi);
}
assert.ok(added.includes('<strong>' + Number(row.predicted).toFixed(1) + ' cycles</strong>'));
assert.ok(Math.abs(row.predicted - sum.rul_max / (1 + Math.exp(-row.score))) < 1e-3, 'the prediction is the score on the curve the sentence describes');

// The same parts along the engine's life: four lines, the chosen cycle marked on each.
const along = P.walkParts(data, at, 1200);
assert.equal(count(along, '<polyline'), 4);
assert.equal(count(along, 'r="5" fill="none"'), 4);
assert.ok(along.includes('>cycle ' + row.end_cycle + '</text>'));

// The largest walks: one row each, one chip per position, and the kind the score's split gives it.
const list = P.topWalks(data);
assert.equal(count(list, '<li class="pg-walk"'), data.top_walks.length);
const groupOf = {};
(sum.groups || []).forEach(g => g.forEach(s => { groupOf[s] = g.join('/'); }));
const stays = data.top_walks.filter(w => { const steps = w.walk.split(' -> '); return steps.slice(1).every(x => x === (groupOf[steps[0]] || steps[0])); }).length;
assert.equal(count(list, '>stays in its ' + home + '</span>'), stays);
assert.equal(count(list, '>crosses to another ' + home + '</span>'), data.top_walks.length - stays);
const layers = data.top_walks[0].walk.split(' -> ').length - 1;
assert.equal(layers, sum.steps);
for (let i = 1; i <= layers; i++) assert.ok(list.includes('<span>After layer ' + i + '</span>'));
assert.equal(count(list, 'class="pg-walk-step"'), data.top_walks.length * (layers + 1));
data.top_walks.forEach((w, i) => {
  assert.ok(list.includes('data-label="average share">' + P.scoreShare(w.mean) + '<'), 'share of walk ' + i);
  assert.ok(list.includes('data-action="walk-show" data-walk="' + i + '"'));
});
assert.ok(list.includes('<strong>' + Number(sum.cancellation_median).toFixed(1) + '\u00d7</strong>'));
assert.ok(list.includes('data-sensor-tip="' + data.top_walks[0].walk.split(' -> ')[0] + '"'));

// The check: two tests, a row per stored curve, this explanation's own ranking marked, and the order it took.
const index = 0, mine = data.flipping.filter(r => Number(r.window) === index);
const check = P.flippingFigure(data, index, 1400);
assert.equal(count(check, 'class="pg-figure pg-check-card"'), 2);
assert.ok(check.includes('>Adding sensors</h4><span class="pg-tag pg-tag--plain">higher is better</span>'));
assert.ok(check.includes('>Removing sensors</h4><span class="pg-tag pg-tag--plain">lower is better</span>'));
assert.equal(count(check, '<polyline'), mine.length);
assert.equal(count(check, '<tr'), mine.length + 2);
assert.equal(count(check, '>this explanation</span>'), 2);
const own = P.ownRanking(sum, mine);
for (const task of ['activation', 'pruning']) {
  const r = mine.find(x => x.task === task && x.ranking === own);
  const names = r.order.split(' ').map(i => sum.sensors[Number(i)]);
  assert.ok(check.includes(names.map(s => '<span class="pg-chip" data-sensor-tip="' + s + '">' + s + '</span>').join('')), task + ' order');
  assert.ok(check.includes('<td class="num">' + Number(r.aufc).toFixed(3) + '</td>'), task + ' value');
  // What the table calls a curve's number is the mean of its points.
  const points = Object.keys(r).filter(k => k.startsWith('step_')).map(k => r[k]);
  assert.ok(Math.abs(points.reduce((a, b) => a + b, 0) / points.length - r.aufc) < 1e-9);
}
for (const a of sum.flipping_aufc) assert.ok(check.includes(Number(a.mean).toFixed(3) + ' ± ' + Number(a.std).toFixed(3)), a.ranking);
// On a narrow panel the two tests stand one under the other.
assert.ok(P.flippingFigure(data, index, 1400).includes('pg-check pg-check--two'));
assert.ok(!P.flippingFigure(data, index, 700).includes('pg-check--two'));
assert.ok(![added, along, list, check].some(text => text.includes('NaN') || text.includes('undefined')));
"""
    done = subprocess.run([node, "-e", script], input=json.dumps(data), text=True, encoding="utf-8",
                          capture_output=True, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr


def _recorded_run(root: Path, name: str, config: dict, summary: dict, per_seed: dict[int, dict]) -> Path:
    """A finished run's record with the settings files train.py writes beside it, one training file per seed."""
    import yaml

    run = root / "reports" / name
    run.mkdir(parents=True)
    (run / "results.csv").write_text("seed,rmse\n42,12.0\n", encoding="utf-8")
    (run / "preprocessing_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"mean": {"rmse": 12.0}, "std": {"rmse": 0.5}, **summary}),
                                      encoding="utf-8")
    for seed, training in per_seed.items():
        (run / f"seed_{seed}").mkdir()
        (run / f"seed_{seed}" / "training_config.yaml").write_text(yaml.safe_dump(training), encoding="utf-8")
    return run


def test_a_listed_run_carries_every_setting_it_recorded_and_none_it_did_not(tmp_path: Path):
    """The list's filters read a run's own files: a value left empty stays, one never written is absent, no default stands in."""
    training = {"epochs": 200, "learning_rate": 0.01, "early_stopping_patience": 20, "lr_schedule": "cosine",
                "optimizer": "adam", "gradient_clip": None, "device": "cuda", "notes": "prose"}
    run = _recorded_run(
        tmp_path, "recorded",
        config={"subset": "FD001", "window_size": 30, "smoothing_alpha": None, "validation_fraction": 0.2,
                "seed": 42, "notes": "prose"},
        summary={"subset": "FD001", "seeds": 3, "graph": "physics", "split": "varies_with_seed", "early_stopping": True,
                 "lr_schedule": "cosine", "temporal_stage": "lstm", "commit": "abc1234", "finished_utc": "20261114T000000Z",
                 "model": {"temporal": "lstm", "temporal_hidden": 32, "gcn_hidden": [32, 64, 32], "gcn_bias": False},
                 "training": {"epochs": 200, "learning_rate": 0.01, "early_stopping_patience": 20, "lr_schedule": "cosine"}},
        per_seed={42: {**training, "seed": 42}, 43: {**training, "seed": 43, "device": "cpu"}})
    _recorded_run(
        tmp_path, "older",
        config={"subset": "FD001", "window_size": 30, "validation_fraction": 0.0},
        summary={"subset": "FD001", "seeds": 5, "graph": "physics", "split": "varies_with_seed", "early_stopping": False,
                 "lr_schedule": "step", "temporal_stage": "lstm", "temporal_hidden": 32, "commit": "0ldc0de-dirty",
                 "finished_utc": "20260909T000000Z"},
        per_seed={})

    runs = {item["name"]: item["parameters"] for item in playground.list_runs(tmp_path)}
    recorded, older = runs["recorded"], runs["older"]

    # What the files hold, under the names the comparison of two runs uses.
    assert recorded["preprocessing.window_size"] == 30 and recorded["preprocessing.seed"] == 42
    assert recorded["model.gcn_hidden"] == [32, 64, 32] and recorded["model.gcn_bias"] is False
    assert (recorded["training.epochs"], recorded["training.learning_rate"]) == (200, 0.01)
    assert (recorded["seeds"], recorded["split"], recorded["early_stopping"]) == (3, "varies_with_seed", True)
    assert (recorded["commit"], recorded["reported_weights"]) == ("abc1234", "best_validation_epoch")
    # Left empty on purpose is a value; prose notes and the seed that differs by design are not settings.
    assert "preprocessing.smoothing_alpha" in recorded and recorded["preprocessing.smoothing_alpha"] is None
    assert "training.gradient_clip" in recorded and recorded["training.gradient_clip"] is None
    assert not {"preprocessing.notes", "training.notes", "training.seed"} & set(recorded)
    # Where the seeds agree the value stands once; where they disagree it is listed seed by seed.
    assert recorded["training.optimizer"] == "adam"
    assert recorded["training.device"] == ["cuda", "cpu"]

    # A run that stored no model and no training settings carries none: no default stands in for them.
    assert not [key for key in older if key.startswith("training.") and key != "training.lr_schedule"]
    assert "model.gcn_hidden" not in older and "model.gcn_bias" not in older
    # What its summary did write is kept: the schedule, the temporal stage and its size.
    assert older["training.lr_schedule"] == "step"
    assert (older["model.temporal"], older["model.temporal_hidden"]) == ("lstm", 32)
    # The two readings the list already made: graph layers unless a run says otherwise, and with
    # nothing held back there is no split and only the last pass to report.
    assert older["model.spatial"] == "gcn"
    assert (older["split"], older["reported_weights"]) == ("none", "final_epoch")

    # A seed that stored no training file answers from the run's summary, which holds the passes
    # but not the update rule: for that seed the rule is not recorded, and nothing is assumed for it.
    import yaml

    (run / "seed_44").mkdir()
    partly = {item["name"]: item["parameters"] for item in playground.list_runs(tmp_path)}["recorded"]
    assert partly["training.epochs"] == 200
    assert partly["training.optimizer"] == ["adam", "adam", "not recorded"]

    # A file that changes is read again, although an unchanged one is read once.
    (run / "seed_43" / "training_config.yaml").write_text(
        yaml.safe_dump({**training, "seed": 43, "device": "cuda", "epochs": 200}), encoding="utf-8")
    (run / "seed_44" / "training_config.yaml").write_text(yaml.safe_dump({**training, "seed": 44}), encoding="utf-8")
    again = {item["name"]: item["parameters"] for item in playground.list_runs(tmp_path)}["recorded"]
    assert (again["training.device"], again["training.optimizer"]) == ("cuda", "adam")


def test_every_stored_run_lists_the_settings_its_own_files_hold():
    """On the runs in reports/: each listed setting is the value in the run's own file, and no file's setting is left out."""
    import yaml

    runs = playground.list_runs(ROOT)
    if not runs:
        pytest.skip("No stored run is in reports/.")
    for run in runs:
        folder, listed = ROOT / "reports" / run["name"], run["parameters"]
        config = yaml.safe_load((folder / "preprocessing_config.yaml").read_text(encoding="utf-8")) \
            if (folder / "preprocessing_config.yaml").is_file() else {}
        for key, value in (config or {}).items():
            if key != "notes":
                assert listed[f"preprocessing.{key}"] == value, (run["name"], key)
        files = sorted(folder.glob("seed_*/training_config.yaml"))
        stored = [yaml.safe_load(path.read_text(encoding="utf-8")) or {} for path in files]
        for key in {key for saved in stored for key in saved} - {"notes", "seed"}:
            values = [saved.get(key) for saved in stored]
            if all(value == values[0] for value in values) and len(files) == len(list(folder.glob("seed_*"))):
                assert listed[f"training.{key}"] == values[0], (run["name"], key)
        if not stored:
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            expected = {f"training.{key}" for key in (summary.get("training") or {}) if key not in ("notes", "seed")}
            if summary.get("lr_schedule") is not None:
                expected.add("training.lr_schedule")
            assert {key for key in listed if key.startswith("training.")} == expected, run["name"]
        assert (listed["seeds"], listed["commit"]) == (run["seeds"], run["commit"])


def test_the_run_list_has_a_filter_for_every_parameter_the_stored_runs_differ_in():
    """More filters: a menu where the runs hold two values or more, the value itself where they agree, and Not recorded as a choice."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the filters of the run list.")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const options = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {window: {location: {hash: ''}}};
vm.createContext(context);
const source = fs.readFileSync('src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {noFilters, runParameters, paramToken, runShown, tidyFilters, filtersInUse, filterBar, moreFilters,
    runFilters, resultRows, setup: function (helpers, state) { T = helpers; S = state; }, state: function () { return S; }};`), context);
const P = context.window.test;
const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const count = (text, part) => text.split(part).length - 1;
const fields = {};
options.fields.forEach(f => { fields[f.key] = f; });

// Four runs: three recorded their settings, the oldest only what its summary holds.
const common = {'preprocessing.subset': 'FD001', 'preprocessing.graph': 'physics', 'preprocessing.window_size': 30,
  'model.temporal': 'lstm', 'model.gcn_hidden': [32, 64, 32], 'training.optimizer': 'adam', 'commit': 'abc1234',
  'reported_weights': 'best_validation_epoch', 'split': 'varies_with_seed', 'seeds': 5};
const run = (name, extra) => ({name, subset: 'FD001', temporal: 'lstm', graph: 'physics', has_weights: true, exploratory: false,
  reported_weights: 'best_validation_epoch', parameters: Object.assign({}, common, extra)});
const runs = [
  run('a', {'preprocessing.smoothing_alpha': 0.5, 'preprocessing.normalization': 'zscore', 'training.learning_rate': 0.001,
    'model.gcn_bias': false, 'training.early_stopping_patience': 20}),
  run('b', {'preprocessing.smoothing_alpha': null, 'preprocessing.normalization': 'zscore', 'training.learning_rate': 0.01,
    'model.gcn_bias': true, 'training.early_stopping_patience': null}),
  run('c', {'preprocessing.smoothing_alpha': 0.5, 'preprocessing.normalization': 'minmax_global', 'training.learning_rate': 0.0001,
    'model.gcn_bias': false, 'training.early_stopping_patience': 50, 'split': 'none', 'seeds': 2,
    'preprocessing.node_feature_mode': 'window'}),
  {name: 'old', subset: 'FD001', temporal: 'lstm', graph: 'physics', has_weights: false, exploratory: false,
    parameters: {'preprocessing.subset': 'FD001', 'preprocessing.graph': 'physics', 'preprocessing.window_size': 30,
      'preprocessing.smoothing_alpha': 0.5, 'preprocessing.normalization': 'zscore', 'model.temporal': 'lstm',
      'commit': '0ldc0de-dirty', 'split': 'varies_with_seed', 'seeds': 5, 'reported_weights': 'best_validation_epoch'}}
];
P.setup({escapeHtml}, {options, fields, runs, jobs: [], resultsTab: 'runs', page: {runs: 0, explanations: 0}, filter: P.noFilters(false)});
const S = P.state();

const params = P.runParameters(runs);
const by = {};
params.forEach(p => { by[p.key] = p; });
// The first row's four are not repeated.
for (const key of ['preprocessing.subset', 'preprocessing.graph', 'model.temporal', 'reported_weights']) assert.ok(!(key in by), key);
// Named as the design form names them, or as a run's Settings part does where the form has no control.
assert.equal(by['training.learning_rate'].label, fields.learning_rate.label);
assert.equal(by['preprocessing.smoothing_alpha'].label, fields.smoothing_alpha.label);
assert.equal(by['seeds'].label, fields.seeds.label);
assert.equal(by['split'].label, 'Held-back engines');
assert.equal(by['training.optimizer'].label, 'Parameter update rule');
assert.equal(by['preprocessing.node_feature_mode'].label, 'Node feature mode');
// In the form's groups and order: how the engines were split stands with the repeats, the commit with the record.
assert.deepEqual([by['preprocessing.normalization'].group, by['model.gcn_bias'].group, by['training.learning_rate'].group,
  by['split'].group].join(' '), 'preprocessing graph training training');
// What only a run's record holds, its commit, is given no filter.
assert.ok(!('commit' in by));
const order = params.map(p => p.key);
assert.equal(order.indexOf('split'), order.indexOf('seeds') + 1);
assert.ok(order.indexOf('preprocessing.normalization') < order.indexOf('model.gcn_bias'));
assert.ok(order.indexOf('model.gcn_bias') < order.indexOf('training.learning_rate'));

// A menu only where the runs that recorded the parameter hold two values or more.
assert.deepEqual(params.filter(p => p.varies).map(p => p.key).sort().join(' '), ['model.gcn_bias',
  'preprocessing.normalization', 'preprocessing.smoothing_alpha', 'seeds', 'split', 'training.early_stopping_patience',
  'training.learning_rate'].sort().join(' '));
assert.ok(!by['preprocessing.window_size'].varies && !by['model.gcn_hidden'].varies && !by['training.optimizer'].varies);
assert.ok(!by['preprocessing.node_feature_mode'].varies);
// The choices, in the form's words, numbers in order, and Not recorded only where a run recorded nothing.
const labels = key => by[key].choices.map(c => c.label).join(' | ');
assert.equal(labels('training.learning_rate'), '0.0001 | 0.001 | 0.01 | Not recorded');
assert.equal(labels('preprocessing.smoothing_alpha'), '0.5 | Off');
assert.equal(labels('model.gcn_bias'), 'Off | On | Not recorded');
assert.equal(labels('training.early_stopping_patience'), '20 | 50 | Off | Not recorded');
assert.equal(labels('split'), 'A different draw for every seed | None held back');
assert.equal(labels('preprocessing.normalization'), [fields.normalization.choices.find(c => c.value === 'minmax_global').label,
  fields.normalization.choices.find(c => c.value === 'zscore').label].sort((a, b) => a.localeCompare(b, 'en')).join(' | '));
assert.equal(by['model.gcn_hidden'].choices[0].label, '32 · 64 · 32');

// Which runs a filter leaves: the value itself, a value left empty, or nothing recorded.
const shown = more => { S.filter = P.noFilters(true); Object.assign(S.filter.more, more); return runs.filter(P.runShown).map(r => r.name).join(''); };
assert.equal(shown({}), 'abcold');
assert.equal(shown({'training.learning_rate': '0.01'}), 'b');
assert.equal(shown({'preprocessing.smoothing_alpha': 'null'}), 'b');
assert.equal(shown({'training.learning_rate': 'unrecorded'}), 'old');
assert.equal(shown({'preprocessing.normalization': '"zscore"', 'model.gcn_bias': 'false'}), 'a');
assert.equal(shown({'seeds': '2', 'split': '"none"'}), 'c');
// With the first row as well.
S.filter = P.noFilters(false); S.filter.files = true; S.filter.more['preprocessing.normalization'] = '"zscore"';
assert.equal(runs.filter(P.runShown).map(r => r.name).join(''), 'ab');
assert.deepEqual(JSON.parse(JSON.stringify(P.filtersInUse())), {first: 1, more: 1});

// The bar: put away, each filter in use is named and can be removed; the count says how many runs are left.
const closed = P.filterBar(params, 2);
assert.ok(closed.includes('aria-expanded="false"') && closed.includes('aria-controls="pg-more"'));
assert.ok(closed.includes('<span class="pg-badge" title="1 filter in use here">1</span>'));
assert.ok(closed.includes(escapeHtml(fields.normalization.label) + ': <strong>' +
  escapeHtml(fields.normalization.choices.find(c => c.value === 'zscore').label) + '</strong>'));
assert.ok(closed.includes('data-action="drop-filter" data-param="preprocessing.normalization"'));
assert.ok(closed.includes('>2 of 4 runs</p>') && closed.includes('data-action="clear-filters"'));
assert.ok(P.moreFilters(params).includes('id="pg-more" aria-label="More filters" hidden>'));
// On show, the menus say it themselves, so no chip repeats them.
S.filter.open = true;
const open = P.filterBar(params, 2), panel = P.moreFilters(params);
assert.ok(open.includes('aria-expanded="true"') && !open.includes('pg-filter-chip'));
assert.ok(!panel.includes(' hidden>'));
assert.equal(count(panel, '<select '), params.filter(p => p.varies).length);
assert.equal(count(panel, 'class="pg-filter is-used"'), 1);
assert.ok(panel.includes('<option value="&quot;zscore&quot;" selected>'));
assert.ok(panel.includes('<option value="unrecorded">Not recorded</option>'));
// A parameter the runs agree on is named with its value, not given a menu.
assert.ok(panel.includes('The same in every stored run that recorded it, so there is nothing to filter by:'));
assert.ok(panel.includes('<dt>' + escapeHtml(fields.window_size.label) + '</dt><dd>30</dd>'));
assert.ok(panel.includes('<dt>Parameter update rule</dt><dd>adam</dd>'));
assert.ok(!panel.includes('data-param="preprocessing.window_size"'));
// Nothing in use: every run counted, nothing to clear, no badge.
S.filter = P.noFilters(false);
const idle = P.filterBar(params, 4);
assert.ok(idle.includes('>4 runs</p>') && !idle.includes('clear-filters') && !idle.includes('pg-badge'));
// A filter of the first row in use is marked on its own menu.
S.filter.subset = 'FD001';
assert.equal(count(P.runFilters(), 'class="pg-filter is-used"'), 1);

// A filter on a value no run holds any more is taken away.
S.filter = P.noFilters(true);
Object.assign(S.filter.more, {'training.learning_rate': '0.5', 'seeds': '2', 'preprocessing.window_size': '30', 'no.such': '1',
  'commit': '"abc1234"'});
P.tidyFilters(params);
assert.deepEqual(JSON.parse(JSON.stringify(S.filter.more)), {'seeds': '2'});

// The State menu's "All" is a value the list understands: every stored run is shown again.
S.filter = P.noFilters(false);
assert.equal(S.filter.state, 'all');
assert.equal(P.resultRows().length, 4);
S.filter.state = 'done';
assert.equal(P.resultRows().length, 4);
S.filter.state = 'live';
assert.equal(P.resultRows().length, 0);
S.filter.state = 'all';
assert.equal(P.resultRows().length, 4);
"""
    done = subprocess.run([node, "-e", script], input=json.dumps(playground.options()), text=True, encoding="utf-8",
                          capture_output=True, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr
