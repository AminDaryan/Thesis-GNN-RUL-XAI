"""One worked example of each function, small enough to read at a glance.

These used to live inside the source files as doctests. They were moved here
when the explanation was moved into the folder README files, so that the code
carries only a one-line description of what each function does. Nothing was
dropped: every example below is the one that used to sit in the docstring of
the function it tests.

The three examples that need PyTorch are in test_examples_torch.py.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.data.labeling import apply_piecewise_rul, piecewise_rul_curve
from thesis_gnn_rul_xai.data.sensors import (
    PAPER_SENSOR_COLUMNS,
    SENSOR_COLS,
    detect_uninformative_sensors,
)
from thesis_gnn_rul_xai.data.smoothing import ewma_series
from thesis_gnn_rul_xai.data.windowing import split_units_by_id
from thesis_gnn_rul_xai.evaluation.metrics import (
    asue,
    evaluate,
    nasa_score,
    prediction_error,
    r_squared,
    rmse,
)
from thesis_gnn_rul_xai.evaluation.published import SUBSETS, published_row
from thesis_gnn_rul_xai.graph.adjacency import (
    adjacency_report,
    build_correlation_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
    pairs_beyond_reach,
)
from thesis_gnn_rul_xai.utils.paths import log_entry_files, project_root
from thesis_gnn_rul_xai.utils.provenance import current_commit, utc_stamp

# --- docsite/ ---------------------------------------------------------


def test_example_engine_overlay_preserves_real_walk_values_and_grouped_paths():
    """The engine and abstract layouts display the same saved FD001 explanation."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    from thesis_gnn_rul_xai.docsite import playground

    root = Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the browser's engine drawing.")
    # A short check made at another setting of the walk rule (explain.py
    # --gamma) is not an explanation these examples read; a full one is.
    stored = next((p for p in sorted((root / "reports").glob("*/explanations/*/summary.json"))
                   if (s := json.loads(p.read_text(encoding="utf-8"))).get("method") == "walks"
                   and s.get("windows", 0) >= 3 and s.get("gamma_setting", "authors") == "authors"), None)
    if stored is None:
        pytest.skip("A stored graph-walk explanation of the real FD001 data is needed.")
    data = playground.explanation_detail(root, stored.parents[2].name, stored.parent.name)
    script = r"""
const fs = require('fs'), assert = require('node:assert/strict');
const E = require('./src/thesis_gnn_rul_xai/docsite/assets/engine-view.js').ThesisEngineView;
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const data = payload.data, sensors = data.summary.sensors || Object.keys(data.summary.mean_abs_start);
const before = JSON.stringify(data), edges = E.edgesFor(data, sensors, []);
const limits = E.peaks(data, sensors, edges), row = data.values[0];
assert.deepEqual(new Set(Object.keys(E.locations)), new Set(sensors));
assert.equal(E.locations.T30.part, E.locations.P30.part);
assert.equal(E.locations.Ps30.part, E.locations.P30.part);
assert.equal(E.locations.NRf.kind, 'derived');
assert.equal(E.locations.BPR.kind, 'derived');
assert.equal(E.locations.W32.kind, 'system');
const opts = {index: 0, graph: payload.graph};
const base = {sensor: null, overlay: true, walk: -1};
const engine = E.graphPicture(data, opts, {...base, layout: 'engine'}, sensors, edges, limits);
const graph = E.graphPicture(data, opts, {...base, layout: 'graph'}, sensors, edges, limits);
const titles = html => [...html.matchAll(/<title>(.*?)<\/title>/g)].map(x => x[1]).sort();
assert.deepEqual(titles(engine), titles(graph));
for (const s of sensors) {
  assert.ok(engine.includes('data-ev-sensor="' + s + '"'));
  assert.ok(engine.includes(row['start_' + s].toFixed(3)));
}
for (const r of data.values) {
  const sensorTotal = sensors.reduce((total, s) => total + r['start_' + s], 0);
  assert.ok(Math.abs(sensorTotal + r.constant - r.score) < 1e-9);
  const cycles = data.summary.rul_max / (1 + Math.exp(-r.score));
  assert.ok(Math.abs(cycles - r.predicted) < 1e-3);
}
for (const [a,b] of edges) {
  assert.ok('link_' + a + '_' + b in row || 'link_' + b + '_' + a in row);
}
for (const a of sensors) for (const b of sensors) {
  if (a !== b && 'link_' + a + '_' + b in row) {
    assert.ok(edges.some(e => e.includes(a) && e.includes(b)));
  }
}
for (const w of data.top_walks) {
  const steps = E.walkSteps(w.walk, sensors), pairs = E.walkPairs(steps, edges);
  assert.equal(steps.length, data.summary.steps + 1);
  assert.deepEqual(steps, w.walk.split(' -> ').map(s => s.split('/')));
  for (const pair of pairs) assert.ok(edges.includes(pair));
  for (let i=0; i<steps.length-1; i++) for (const a of steps[i]) for (const b of steps[i+1]) {
    const edge = edges.find(e => e.includes(a) && e.includes(b));
    if (a !== b && edge) assert.ok(pairs.includes(edge));
  }
}
assert.equal(JSON.stringify(data), before);
assert.equal(limits.nodes, Math.max(...data.values.flatMap(r => sensors.map(s => Math.abs(r['start_' + s])))));
console.log(JSON.stringify({edges: edges.length, sensors: sensors.length, nodePeak: limits.nodes}));
"""
    result = subprocess.run([node, "-e", script], cwd=root,
                            input=json.dumps({"data": data, "graph": playground.options()["graph"]}),
                            text=True, encoding="utf-8", capture_output=True, check=True)
    measured = json.loads(result.stdout)
    named = data["summary"].get("sensors") or list(data["summary"]["mean_abs_start"])
    assert measured["sensors"] == len(named)
    assert measured["nodePeak"] == max(abs(r[f"start_{s}"]) for r in data["values"]
                                       for s in named)


@pytest.mark.parametrize("scope", ["engine", "test"])
def test_example_shared_selection_updates_on_input_without_replacing_controls(scope):
    """Dragging through saved FD001 results updates sensor details before release."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    from thesis_gnn_rul_xai.docsite import playground

    root = Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to check the browser's live selection.")
    # At least three windows, since the slider moves to the third; and not a
    # short check made at another setting of the walk rule (explain.py --gamma).
    stored = next((p for p in sorted((root / "reports").glob("*/explanations/*/summary.json"))
                   if (s := json.loads(p.read_text(encoding="utf-8"))).get("method") == "walks"
                   and s.get("scope") == scope and s.get("windows", 0) >= 3
                   and s.get("gamma_setting", "authors") == "authors"), None)
    if stored is None:
        pytest.skip("A stored graph-walk explanation of the real FD001 data is needed.")
    data = playground.explanation_detail(root, stored.parents[2].name, stored.parent.name)
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const payload = JSON.parse(fs.readFileSync(0, 'utf8')), data = payload.data;
const E = require('./src/thesis_gnn_rul_xai/docsite/assets/engine-view.js').ThesisEngineView;
const frames = new Map(); let serial = 0;
const context = {window: {
  requestAnimationFrame(fn) { frames.set(++serial, fn); return serial; },
  cancelAnimationFrame(id) { frames.delete(id); }
}};
vm.createContext(context);
const source = fs.readFileSync('./src/thesis_gnn_rul_xai/docsite/assets/playground.js', 'utf8');
vm.runInContext(source.replace('window.thesisPlayground = start;', `
  window.test = {S, onInput, selectExplanationIndex, explanationSelection, wireExplanationSelection,
    setup: function (element, helpers) { root = element; T = helpers; }};`), context);
const P = context.window.test;
const range = {type: 'range', value: '0', attributes: {}, style: {setProperty() {}},
  hasAttribute(name) { return name === 'data-explain-index'; },
  setAttribute(name, value) { this.attributes[name] = value; }};
const pick = {value: '0'};
const fields = new Map();
const selection = {querySelectorAll() { return [range, pick]; },
  querySelector(key) { if (!fields.has(key)) fields.set(key, {}); return fields.get(key); }};
const example = {innerHTML: ''};
const root = {querySelector(key) { return key === '.pg-selection' ? selection :
  key === '[data-walk-example]' ? example : null; },
  set innerHTML(_) { assert.fail('Dragging must not replace the shared selector or open details.'); }};
P.setup(root, {escapeHtml: String, fixed: (x, places) => Number(x).toFixed(places)});
P.S.tab = 'explain'; P.S.explain = {values: data, index: 0, chart: 'engine'};
const state = {sensor: 'T30', layout: 'engine', overlay: true, walk: -1};
const regions = {'.ev-scroll': {innerHTML: '', scrollLeft: 120}, '.ev-details': {innerHTML: ''}};
let mounts = 0;
const container = {set innerHTML(_) { mounts++; }, querySelector(key) { return regions[key]; }};
P.S.explain.engineController = E.mount(container, data, {index: 0, edges: null, graph: payload.graph}, state);
P.wireExplanationSelection(root, data, [], 800);
const flush = () => { const callbacks = [...frames.values()]; frames.clear(); callbacks.forEach(f => f()); };
const html = P.explanationSelection(data);
assert.ok(html.includes(data.summary.scope === 'engine' ? 'Engine cycle' : 'Test engine'));
for (const [i, row] of data.values.entries()) {
  const name = data.summary.scope === 'engine' ? 'Cycle ' + row.end_cycle : 'Engine ' + row.unit;
  assert.ok(html.includes('value="' + i + '"' + (i === 0 ? ' selected' : '') + '>' + name));
}
assert.equal(fields.get('[data-action="explain-previous"]').disabled, true);
for (let i = 1; i <= 2; i++) { range.value = String(i); P.onInput({target: range}); }
assert.equal(frames.size, 1);
flush(); // No change or pointer-release event: input alone must update the view.
assert.equal(P.S.explain.index, 2);
assert.equal(range.value, '2'); assert.equal(pick.value, '2');
const row = data.values[2];
assert.ok(regions['.ev-details'].innerHTML.includes(row.start_T30.toFixed(3)));
assert.ok(regions['.ev-scroll'].innerHTML.includes(row.start_T30.toFixed(3)));
assert.ok(example.innerHTML.includes('cycle ' + row.end_cycle));
assert.equal(fields.get('[data-selection-predicted]').textContent, row.predicted.toFixed(1));
assert.equal(state.sensor, 'T30'); assert.equal(state.overlay, true);
assert.equal(regions['.ev-scroll'].scrollLeft, 120); assert.equal(mounts, 1);
state.layout = 'graph'; range.value = '1'; P.onInput({target: range}); flush();
assert.equal(state.layout, 'graph'); assert.equal(state.sensor, 'T30');
assert.ok(regions['.ev-details'].innerHTML.includes(data.values[1].start_T30.toFixed(3)));
P.selectExplanationIndex(data.values.length + 1); flush();
assert.equal(P.S.explain.index, data.values.length - 1);
assert.equal(fields.get('[data-action="explain-next"]').disabled, true);
P.selectExplanationIndex(-1); flush();
assert.equal(P.S.explain.index, 0);
assert.equal(fields.get('[data-action="explain-previous"]').disabled, true);
const lastPicture = regions['.ev-scroll'].innerHTML;
P.selectExplanationIndex(1); P.S.tab = 'results'; flush();
assert.equal(regions['.ev-scroll'].innerHTML, lastPicture);
console.log('Live input, saved identifiers, retained selection and boundary controls passed.');
"""
    result = subprocess.run([node, "-e", script], cwd=root,
                            input=json.dumps({"data": data, "graph": playground.options()["graph"]}), text=True,
                            encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changes", [{}, {"smoothing_alpha": 0.25},
                                    {"smoothing": False, "window_size": 25, "window_stride": 2,
                                        "validation_fraction": 0.0, "rul_max": 150, "seed": 7}])
def test_example_preprocessing_story(loaded, changes):
    """The diagram's rows, changed readings and overlapping windows come from the real FD001 files."""
    from thesis_gnn_rul_xai.docsite import playground

    settings = playground.validate_settings(changes)
    answer = playground.stage_detail("preprocessing", settings)
    opening = answer["steps"][0]
    assert opening["handoff"] == "20,631 rows · 27 columns"
    assert opening["name"] == "Engine histories and remaining-life answers"
    assert opening["example"]["source"] == "train_FD001.txt"
    assert opening["example"]["label_state"] == "uncapped"
    assert opening["summary"] == ""
    assert "uncapped remaining-life label" in opening["columns"]
    sample = opening["example"]["rows"]
    assert sample == [{"engine": 1, "cycle": 1, "reading": 641.82, "life": 191},
                      {"engine": 1, "cycle": 2, "reading": 642.15, "life": 190},
                      {"engine": 1, "cycle": 3, "reading": 642.35, "life": 189}]
    assert sample[1]["reading"] == loaded.train.loc[1, "sensor_2"]
    stages = {step.get("picture"): step for step in answer["steps"]}
    assert stages["cap"]["example"]["after"] == f"{min(190, settings['rul_max'])} cycles"
    if settings["smoothing"]:
        alpha = settings["smoothing_alpha"]
        expected = alpha * 642.15 + (1 - alpha) * 641.82
        assert float(stages["smooth"]["example"]["after"]) == pytest.approx(expected)
        assert stages["smooth"]["example"]["note"] == (
            f"{alpha:g} x current reading + {1 - alpha:g} x previous smoothed reading")
        assert stages["rescale"]["example"]["before"] == f"{expected:.3f}"
        if alpha == 0.5:
            assert stages["rescale"]["example"]["after"] == "-1.6039"
    else:
        assert "smooth" not in stages
        assert stages["rescale"]["example"]["before"] == "642.15"
    groups = stages["hold"]["example"]["items"]
    assert sum(int(group["detail"].split()[0].replace(",", "")) for group in groups) == len(loaded.train)
    windows = stages["window"]["example"]["windows"]
    for index, item in enumerate(windows):
        start = 1 + index * settings["window_stride"]
        end = start + settings["window_size"] - 1
        assert item == {"start": start, "end": end, "life": min(192 - end, settings["rul_max"])}


def test_example_rul_chart_uses_file_cycles_and_true_uncapped_life(loaded):
    """Both plotted lines share the file's cycle coordinates and the same vertical scale."""
    import xml.etree.ElementTree as ET

    from thesis_gnn_rul_xai.docsite.figures import draw_rul_curve

    root = ET.fromstring(draw_rul_curve())
    curves = {node.attrib["data-curve"]: node for node in root.iter() if "data-curve" in node.attrib}
    raw_points = [[float(v) for v in p.split(",")] for p in curves["raw"].attrib["points"].split()]
    capped = [[float(v) for v in p.split(",")] for p in curves["capped"].attrib["points"].split()]
    assert len(capped) == 192
    assert raw_points[0] == [90.0, 60.0]
    assert raw_points[-1] == capped[-1] == [790.0, 360.0]
    assert capped[0][1] > raw_points[0][1]
    assert capped[66][1] == capped[0][1]
    assert capped[67][1] > capped[66][1]


def test_example_preprocessing_without_files(monkeypatch):
    """Missing NASA files produce descriptions without invented readings or counts."""
    from thesis_gnn_rul_xai.docsite import playground

    monkeypatch.setattr(playground.design, "_files_present", lambda subset: False)
    answer = playground.stage_detail("preprocessing", playground.validate_settings({}))
    assert "not in data/raw/" in answer["note"]
    assert all("example" not in step for step in answer["steps"])
    assert answer["steps"][0]["handoff"] == "Rows · 27 columns"


# --- data/labeling/ ---------------------------------------------------


def test_example_design_data_tutorial(loaded, monkeypatch):
    """The Data page reproduces every cell of its complete FD001 sample tables."""
    namespace = _check_design_tutorial("01_the_problem.md", monkeypatch)
    assert namespace["training"].shape == (20631, 26)
    assert namespace["labelled"].shape == (20631, 27)
    assert namespace["history"].shape == (13096, 26)
    assert namespace["test_labelled"].shape == (13096, 27)
    assert namespace["selected"].shape == (20631, 14)
    pd.testing.assert_frame_equal(
        namespace["selection_output"], namespace["selection_input"][list(PAPER_SENSOR_COLUMNS)],
    )


def test_example_design_preprocessing_tutorial(loaded, monkeypatch):
    """The preparation examples preserve rows and reproduce every displayed cell."""
    namespace = _check_design_tutorial("04_the_pipeline.md", monkeypatch)
    for preceding, following in (
        ("cap_output", "smooth_input"), ("smooth_output", "condition_input"),
        ("condition_output", "scale_input"),
    ):
        pd.testing.assert_frame_equal(namespace[preceding], namespace[following])
    assert namespace["cap_output"].rul.tolist() == [125, 125, 125]
    np.testing.assert_allclose(namespace["smooth_output"].sensor_2, [641.82, 641.985, 642.1675])
    raw, smooth = (namespace[name].query("unit == 1").set_index("cycle").sensor_4
                   for name in ("capped", "smoothed"))
    assert (raw.head(30).mean(), raw.tail(30).mean()) == pytest.approx((1400.107, 1421.924), abs=5e-4)
    np.testing.assert_allclose(raw.loc[125:128], [1406.42, 1418.50, 1399.49, 1407.06])
    np.testing.assert_allclose(smooth.loc[125:128], [1405.62175, 1412.06088, 1405.77544, 1406.41772], atol=5e-6)
    assert 0.5 * raw[127] + 0.5 * smooth[126] == pytest.approx(smooth[127])
    steps = [namespace[name].groupby("unit").sensor_4.diff().abs().mean() for name in ("capped", "smoothed")]
    assert steps == pytest.approx([4.557, 1.866], abs=5e-4)
    assert [raw.diff().abs().mean(), smooth.diff().abs().mean()] == pytest.approx([4.155, 1.689], abs=5e-4)
    stats = namespace["scale_statistics"].query("column == 'sensor_2'").iloc[0]
    assert stats["mean"] == pytest.approx(642.6751709174424)
    assert stats["std"] == pytest.approx(0.4303079837962261)
    assert namespace["scale_output"].sensor_2.iloc[1] == pytest.approx(-1.603900, abs=5e-7)
    assert namespace["windows"].features.shape == (3, 14, 3)
    for i in range(3):
        expected = namespace["window_input"].iloc[i:i + 3][list(PAPER_SENSOR_COLUMNS)].to_numpy()
        np.testing.assert_allclose(namespace["windows"].features[i].T, expected)
    assert len(namespace["mini_kept"]) == len(namespace["mini_held"]) == 1
    for prefix in ("sets", "two_sets"):
        pd.testing.assert_frame_equal(
            namespace[f"{prefix}_input"], namespace[f"{prefix}_output"].drop(columns="destination"),
        )
    assert namespace["sets_output"].destination.tolist() == ["train"] * 3 + ["validation"] * 3 + ["test"] * 3
    assert namespace["two_sets_output"].destination.tolist() == ["train"] * 6 + ["test"] * 3
    assert len(namespace["prepared"].train) + len(namespace["prepared"].validation) == 17731


def test_example_design_graph_tutorial(loaded, monkeypatch):
    """The graph tutorial reproduces every entry of both complete connection tables."""
    namespace = _check_design_tutorial("02_graphs.md", monkeypatch, minimum_tables=8)
    assert namespace["sample"].shape == (3, 15)
    assert namespace["raw_graph"].shape == namespace["scaled_output"].shape == (14, 15)


def _check_design_tutorial(filename, monkeypatch, minimum_tables=10, folder="learn"):
    """Execute the page in order and compare its named samples with every printed cell."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    document = (root / "docs" / folder / filename).read_text(encoding="utf-8")
    namespace = {}
    blocks = re.findall(r"```python\n(.*?)\n```", document, flags=re.DOTALL)
    assert blocks
    for code in blocks:
        exec(compile(code, filename, "exec"), namespace)
    tables = re.findall(r"<!-- sample: (\w+) -->\n((?:\|[^\n]+\n)+)", document)
    assert len(tables) >= minimum_tables
    for name, table in tables:
        lines = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                 for line in table.strip().splitlines()]
        actual = namespace[name]
        assert lines[0] == list(actual.columns), name
        assert len(lines) - 2 == len(actual), name
        for expected, row in zip(lines[2:], actual.itertuples(index=False, name=None), strict=True):
            assert len(expected) == len(row), name
            for printed, value in zip(expected, row, strict=True):
                if isinstance(value, (int, float)):
                    assert float(printed) == pytest.approx(value, abs=5.1e-7, rel=0), (name, printed, value)
                else:
                    assert printed == str(value), (name, printed, value)
    if filename == "01_the_problem.md":
        first_lines = (root / "data" / "raw" / "train_FD001.txt").read_text().splitlines()[:3]
        for line in first_lines:
            assert line.rstrip() in document
    return namespace


def test_example_apply_piecewise_rul():
    """An engine with 300 cycles left is labelled 125; one with 47 keeps 47."""
    frame = pd.DataFrame({"rul": [300, 130, 125, 47, 0]})
    capped = apply_piecewise_rul(frame, rul_max=125)["rul"].tolist()
    assert capped == [125, 125, 125, 47, 0]


def test_example_piecewise_rul_curve():
    """A six-cycle engine with a ceiling of 3 is flat, then counts down."""
    assert piecewise_rul_curve(6, rul_max=3).tolist() == [3, 3, 3, 2, 1, 0]


# --- data/sensors/ ----------------------------------------------------


def test_example_detect_uninformative_sensors():
    """Sensor 1 is a flat line and sensor 2 varies, so only sensor 2 is kept."""
    frame = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in SENSOR_COLS})
    frame["sensor_2"] = [1.0, 2.0, 3.0]

    report = detect_uninformative_sensors(frame)

    assert report["informative"] == ["T24"]
    assert "T2" in report["constant"]


# --- data/smoothing/ --------------------------------------------------


def test_example_ewma_series():
    """Half of each new reading, half of the smoothed value before it."""
    assert ewma_series([0.0, 1.0, 1.0], alpha=0.5).tolist() == [0.0, 0.5, 0.75]


# --- data/windowing/ --------------------------------------------------


def test_example_split_units_by_id():
    """Ten engines, one fifth held back: eight and two, sharing nothing."""
    train, held_back = split_units_by_id(
        range(1, 11), validation_fraction=0.2, seed=0
    )

    assert (len(train), len(held_back)) == (8, 2)
    assert set(train) & set(held_back) == set()


# --- evaluation/metrics.py ----------------------------------------------


def test_example_prediction_error():
    """Ten cycles short, then ten cycles long."""
    assert prediction_error([100, 50], [90, 60]).tolist() == [-10.0, 10.0]


def test_example_rmse():
    """Two errors of ten cycles and one of zero."""
    assert round(rmse([100, 50, 20], [110, 40, 20]), 4) == 8.165


def test_example_nasa_score():
    """Ten cycles late costs more than ten cycles early; perfect scores zero."""
    assert round(nasa_score([100], [110]), 4) == 1.7183
    assert round(nasa_score([100], [90]), 4) == 1.1581
    assert nasa_score([100, 50], [100, 50]) == 0.0


def test_example_r_squared():
    """A perfect fit scores 1; always predicting the mean scores 0."""
    assert r_squared([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert round(r_squared([1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5]), 4) == 0.0


def test_example_asue():
    """Only the two pessimistic engines count: (10 + 20) / 2; with none, not a number."""
    assert asue([100, 50, 30], [90, 60, 10]) == 15.0
    # Assumption A-39: no engine predicted early is 0 / 0 in Equation 24.
    with pytest.warns(RuntimeWarning):
        assert np.isnan(asue([100], [110]))


def test_example_evaluate():
    """All four measures at once, plus how many engines were overestimated."""
    result = evaluate([100, 50, 30], [95, 55, 30])

    assert round(result["rmse"], 4) == 4.0825
    assert result["n_late"] == 2


# --- evaluation/published.py --------------------------------------------


def test_example_published_row_prints_only_what_the_table_prints():
    """Table 11 gives the model without its LSTM stage an RMSE and nothing else.

    Its average across the four sub-datasets and its difference from the full
    model are the two figures Table 11 prints beside it.
    """
    without = published_row("without_lstm", "FD001")
    assert (without.rmse, without.score, without.r2, without.asue) == (14.93, None, None, None)

    rows = [published_row("without_lstm", subset).rmse for subset in SUBSETS]
    assert rows == [14.93, 23.34, 16.25, 24.21]
    assert round(sum(rows) / 4, 2) == 19.68
    assert round(without.rmse - published_row("full", "FD001").rmse, 2) == 3.15

    with pytest.raises(ValueError, match="no published row called 'transformer'"):
        published_row("transformer", "FD001")


# --- utils/paths.py -----------------------------------------------------


def test_example_log_entry_files_lists_the_assumptions_then_the_findings():
    """Every A-NN.md in number order, then every F-NN.md; the folders' README files are not entries."""
    names = [path.name for path in log_entry_files()]
    assumptions = [name for name in names if name.startswith("A-")]
    assert names[0] == "A-01.md"
    assert names[len(assumptions)] == "F-01.md"
    assert names == sorted(assumptions) + sorted(set(names) - set(assumptions))
    assert "README.md" not in names
    assert (project_root() / "docs" / "assumptions" / "README.md").is_file()


# --- utils/provenance.py ------------------------------------------------


def test_example_provenance_stamps_have_the_form_folder_names_use():
    """A short commit, marked -dirty when the files differ from it; UTC to the second."""
    assert re.fullmatch(r"[0-9a-f]{7,}(-dirty)?|unknown", current_commit())
    stamp = utc_stamp()
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    assert ":" not in stamp


# --- graph/adjacency.py -------------------------------------------------


def test_example_normalize_adjacency():
    """Two sensors joined by one edge: every entry becomes 1 / sqrt(2 * 2)."""
    result = normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))

    assert result.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_example_normalize_adjacency_row_form_differs():
    """The two forms agree only while every sensor has the same number of
    neighbours. On a star they part company: the rows of the row form always
    add to 1, and the rows of the other form do not."""
    star = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    by_row = normalize_adjacency(star, method="row").sum(axis=1)
    symmetric = normalize_adjacency(star).sum(axis=1)

    assert [round(float(value), 6) for value in by_row] == [1.0, 1.0, 1.0]
    assert [round(float(value), 3) for value in symmetric] == [1.15, 0.908, 0.908]


def test_example_correlation_graph_keeps_the_strongest_pairs():
    """x and y rise together exactly (correlation 1); z moves against both at
    -0.4. Asked for one edge, x-y is kept. Asked for two, the tie between x-z
    and y-z goes to x-z, the pair that comes first in column order."""
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],
        "z": [4.0, 1.0, 3.0, 2.0],
    })
    assert frame.corr().loc["x", "z"] == pytest.approx(-0.4)

    one, _ = build_correlation_adjacency(frame, ["x", "y", "z"], n_edges=1, symbols="xyz")
    two, _ = build_correlation_adjacency(frame, ["x", "y", "z"], n_edges=2, symbols="xyz")
    cut, _ = build_correlation_adjacency(frame, ["x", "y", "z"], threshold=0.39, symbols="xyz")

    assert one.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert two.tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    assert cut.tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]


def test_example_adjacency_report():
    """The paper's own sensor graph, measured."""
    matrix, symbols = build_physics_adjacency()

    report = adjacency_report(matrix, symbols)

    assert (report["n_edges"], report["diameter"]) == (41, 4)
    assert report["articulation_points"] == ["T24"]


def test_example_pairs_beyond_reach():
    """Three layers leave four pairs of sensors unable to reach each other."""
    matrix, symbols = build_physics_adjacency()

    assert pairs_beyond_reach(matrix, symbols, n_layers=3) == [
        ("BPR", "W32", 4),
        ("NRf", "W32", 4),
        ("Nf", "W32", 4),
        ("P15", "W32", 4),
    ]
    assert pairs_beyond_reach(matrix, symbols, n_layers=4) == []



# --- docs/resources/: the papers the project builds on ---------------


def test_example_resources_cmapss(raw_dir, monkeypatch):
    """The C-MAPSS Resources page reproduces every printed cell from all four subsets."""
    from thesis_gnn_rul_xai.data.loading import EXPECTED_SHAPES

    names = [f"{kind}_{subset}.txt" for subset in ("FD001", "FD002", "FD003", "FD004")
             for kind in ("train", "test", "RUL")]
    missing = [name for name in names if not (raw_dir / name).is_file()]
    if missing:
        pytest.skip(f"The C-MAPSS page reads all four subsets; missing from {raw_dir}: {missing}")
    namespace = _check_design_tutorial("cmapss.md", monkeypatch, minimum_tables=12, folder="resources")
    counts = namespace["subset_counts"].set_index("subset")
    for subset, expected in EXPECTED_SHAPES.items():
        assert counts.loc[subset, "training engines (files)"] == expected["train_units"]
        assert counts.loc[subset, "test engines (files)"] == expected["test_units"]
    assert (counts.loc["FD004", "training engines (readme)"],
            counts.loc["FD004", "test engines (readme)"]) == (248, 249)
    conditions = namespace["flight_conditions"]
    assert len(conditions) == 6
    assert (conditions["sensor 18 in every row"] == conditions["guide fan speed"]).all()
    totals = namespace["score_totals"].set_index("engines")
    assert totals.loc["all 100", "this project (nasa_score)"] < totals.loc["all 100", "Eq. (11) as printed"]
    assert totals.loc["the other 99", "this project (nasa_score)"] > totals.loc["the other 99", "Eq. (11) as printed"]
    at_fifty = namespace["at_fifty"].set_index("error d")
    assert at_fifty.loc[50, "late ÷ 10, early ÷ 13"] == pytest.approx(np.expm1(5.0))
    assert at_fifty.loc[-50, "late ÷ 10, early ÷ 13"] == pytest.approx(np.expm1(50 / 13))


def test_example_gcn_resource(loaded, monkeypatch):
    """The GCN page reproduces every table and its stability argument on the Physics graph."""
    namespace = _check_design_tutorial("gcn.md", monkeypatch, minimum_tables=8, folder="resources")
    eigenvalues = namespace["eigenvalues"]
    # Kipf and Welling, Section 2.2: the matrix of Eq. 7 has eigenvalues in [0, 2].
    assert eigenvalues["Before the trick"].max() == pytest.approx(2.0)
    assert eigenvalues["Before the trick"].min() >= 0
    # After the renormalisation trick they lie in (-1, 1].
    assert eigenvalues["After the trick"].max() == pytest.approx(1.0)
    assert eigenvalues["After the trick"].min() > -1
    sizes = namespace["sizes"].set_index("Applications")
    assert sizes.loc[10, "Before the trick"] > 100 * sizes.loc[0, "Before the trick"]
    assert (sizes["After the trick"].diff().dropna() < 0).all()
    stretch = namespace["stretch"].set_index("Applications")
    assert (stretch["After the trick"] <= 1 + 1e-12).all()
    # The row form shares the eigenvalues but is not symmetric, so it can lengthen some lists.
    assert (stretch["Row form"] > 1).all()
    assert namespace["growth"].max() < 1
    assert namespace["pieces"]["Eigenvectors"].sum() == 14


def test_example_resource_shap(loaded, monkeypatch):
    """The SHAP resource page: exact, kernel-regression and Linear SHAP values agree on engine 5."""
    namespace = _check_design_tutorial("shap.md", monkeypatch, minimum_tables=23, folder="resources")
    f, x, readings = namespace["f"], namespace["x"], namespace["readings"]
    phi_mean, phi_line = namespace["phi_mean"], namespace["phi_line"]
    # Local accuracy under both rules for a missing reading: the values add to f(x) - E[f(X)].
    assert phi_mean.sum() == pytest.approx(f(x) - f(readings).mean(), abs=1e-9)
    assert phi_line.sum() == pytest.approx(f(x) - f(readings).mean(), abs=1e-9)
    # Theorem 2 and Corollary 1 on real FD001 numbers.
    np.testing.assert_allclose(namespace["kernel_line"]["Regression, Shapley kernel"], phi_line, atol=1e-9)
    np.testing.assert_allclose(namespace["linear_shap"]["w (x - E[x])"], phi_mean, atol=1e-9)
    # A one-switch weighted least-squares slope is T50's Shapley value (difference of two weighted means).
    assert namespace["one_switch"][1] == pytest.approx(phi_line[namespace["t50"]], abs=1e-9)
    # Equal weights break consistency by exactly 1/16 cycle; the Shapley kernel does not move.
    change = namespace["consistency"].set_index("Method")["Change"]
    assert change["Regression, equal weights"] == pytest.approx(-0.0625, abs=1e-9)
    assert abs(change["Regression, Shapley kernel"]) < 1e-9
    # Leave one out misses local accuracy under the line fill.
    assert sum(namespace["leave_one_out"]) != pytest.approx(phi_line.sum(), abs=1e-3)
    # Figure 4 of Lundberg and Lee, worked exactly with Eq. 4.
    assert namespace["figure_4"]["Shapley value"].tolist() == pytest.approx([1, 1, 0, 3, 2, 0])


# --- data/normalization/README.md: counting the operating conditions -------


def test_example_condition_count_rests_on_altitude(raw_dir):
    """Rounding merges the Mach numbers; altitude alone keeps FD002's and FD004's six conditions apart."""
    from thesis_gnn_rul_xai.data.loading import read_cmapss_frame
    from thesis_gnn_rul_xai.data.normalization import OperatingConditionEncoder
    from thesis_gnn_rul_xai.data.sensors import OP_SETTING_COLS

    names = [f"{kind}_{subset}.txt" for subset in ("FD002", "FD004") for kind in ("train", "test")]
    missing = [name for name in names if not (raw_dir / name).is_file()]
    if missing:
        pytest.skip(f"Needs the FD002 and FD004 files; missing from {raw_dir}: {missing}")
    for name in names:
        frame = read_cmapss_frame(raw_dir / name)
        settings = frame[list(OP_SETTING_COLS)].to_numpy(dtype=float)
        # The README's table: altitude rounded, Mach to two decimals, throttle angle.
        conditions = np.unique(
            np.column_stack([np.round(settings[:, 0]), np.round(settings[:, 1], 2), settings[:, 2]]),
            axis=0,
        )
        np.testing.assert_allclose(
            conditions,
            [[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100], [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]],
        )
        # What the encoder counts: all three rounded, which leaves the Mach number only 0 and 1.
        decimals = OperatingConditionEncoder().round_decimals
        assert decimals == 0
        rounded = np.unique(np.round(settings, decimals), axis=0)
        assert rounded.tolist() == [[0, 0, 100], [10, 0, 100], [20, 1, 100],
                                    [25, 1, 60], [35, 1, 100], [42, 1, 100]]
        # Mach number and throttle angle, rounded, give three combinations; altitude alone gives six.
        assert np.unique(np.round(settings[:, 1:], decimals), axis=0).tolist() == [[0, 100], [1, 60], [1, 100]]
        assert np.unique(np.round(settings[:, 0], decimals)).tolist() == [0, 10, 20, 25, 35, 42]
        if name.startswith("train"):
            assert OperatingConditionEncoder().fit(frame).n_detected == 6


# --- what the measures and the explanations do and do not guarantee --------


def test_example_capping_guarantees_only_rmse_and_the_score(raw_dir):
    """Capping the FD001 test labels cannot raise RMSE or the NASA score; R-squared and ASUE can worsen."""
    # Assumption A-02, docs/assumptions/A-02.md, corrected 2026-10-02.
    labels = np.loadtxt(raw_dir / "RUL_FD001.txt")
    capped = np.minimum(labels, 125)
    assert int((labels > 125).sum()) == 11

    # Engine 25 has 145 cycles left. Any prediction p <= 125 is early, and
    # capping takes exactly 20 cycles off that early error.
    assert labels[24] == 145
    for p in (0.0, 110.0, 125.0):
        before = prediction_error([labels[24]], [p])[0]
        after = prediction_error([capped[24]], [p])[0]
        assert before < 0 and after <= 0 and abs(before) - abs(after) == 20

    # R-squared: predicting the mean uncapped label for every engine scores 0
    # uncapped and below 0 capped, because capping shrinks the labels' spread.
    constant = np.full(100, labels.mean())
    assert labels.mean() == pytest.approx(75.52)
    assert np.sum((labels - labels.mean()) ** 2) == pytest.approx(172686.96)
    assert np.sum((capped - capped.mean()) ** 2) == pytest.approx(160586.75)
    assert r_squared(labels, constant) == pytest.approx(0.0, abs=1e-12)
    assert r_squared(capped, constant) == pytest.approx(-0.000713, abs=5e-7)
    assert rmse(capped, constant) < rmse(labels, constant)
    assert nasa_score(capped, constant) < nasa_score(labels, constant)

    # ASUE: engine 74 (126 cycles) predicted exactly 125 and engine 1 (112)
    # predicted 100. Capped, engine 74 has no shortfall and leaves the average,
    # and engine 1, below the ceiling, keeps its shortfall: the average rises.
    pair = labels[[73, 0]]
    assert pair.tolist() == [126.0, 112.0]
    assert asue(pair, [125.0, 100.0]) == 6.5
    assert asue(np.minimum(pair, 125), [125.0, 100.0]) == 12.0
    # Engine 25 (145) predicted 123.5 in engine 1's place: its shortfall
    # shrinks by 20 while engine 74 leaves, and the average falls.
    pair = labels[[73, 24]]
    assert pair.tolist() == [126.0, 145.0]
    assert asue(pair, [125.0, 123.5]) == 11.25
    assert asue(np.minimum(pair, 125), [125.0, 123.5]) == 1.5


def test_example_stored_predictions_obey_the_two_capping_guarantees(raw_dir):
    """Stored FD001 predictions under a ceiling of 125: capping never raises RMSE or the score."""
    # ASUE cannot rise either while no prediction reaches 125 (assumption A-02).
    import yaml

    from thesis_gnn_rul_xai.utils.paths import reports_dir

    root = reports_dir()
    files = sorted(root.glob("fd001_*/seed_*/test_predictions.csv")) if root.is_dir() else []
    labels = np.loadtxt(raw_dir / "RUL_FD001.txt")
    capped = np.minimum(labels, 125)
    checked = 0
    for file in files:
        settings_file = file.parents[1] / "preprocessing_config.yaml"
        if not settings_file.is_file():
            continue
        settings = yaml.safe_load(settings_file.read_text(encoding="utf-8")) or {}
        if settings.get("rul_max") != 125:
            continue
        saved = pd.read_csv(file)
        if saved["unit"].tolist() != list(range(1, 101)):
            continue
        guess = saved["predicted"].to_numpy(dtype=float)
        assert guess.max() <= 125, file
        assert rmse(capped, guess) <= rmse(labels, guess), file
        assert nasa_score(capped, guess) <= nasa_score(labels, guess), file
        if guess.max() < 125:
            assert asue(capped, guess) <= asue(labels, guess), file
        checked += 1
    if not checked:
        pytest.skip("no stored FD001 test predictions in reports/")


def test_example_a_different_constant_breaks_the_adding_up_or_the_order(loaded):
    """Measuring stored Shapley values from the mean label breaks either the adding-up or the order by size."""
    # Assumption A-24, docs/assumptions/A-24.md, corrected 2026-10-02.
    import json
    from pathlib import Path

    from thesis_gnn_rul_xai.config import load_config
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.utils.paths import reports_dir

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "fd001.yaml")
    prepared = run_preprocessing(config)
    labels = np.concatenate([prepared.train.targets, prepared.validation.targets])
    assert len(labels) == 17731
    mean_label = float(labels.mean())
    assert mean_label == pytest.approx(80.638, abs=5e-4)
    # Runs prepared with these settings share these labels. They are picked by
    # the settings each run saved, not by the digest in its folder name: until
    # 2026-10-03 that digest also hashed the prose notes, so runs with these
    # settings carry cfgba82e3 or cfg72f506 depending on their notes.
    digest = config.digest()
    named = ("fd001_physics_row_cap_5seed_varsplit_cfgba82e3_5602773-dirty_20260914T233911Z",
             "shap_engine_unit5_seed42_c2bf7d4-dirty_20260916T070850Z")

    folder = reports_dir()
    tables = sorted(folder.glob("fd001_*/explanations/shap_*/values.csv")) if folder.is_dir() else []
    checked = 0
    for path in tables:
        summary_file = path.parent / "summary.json"
        if not summary_file.is_file():
            continue
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("method") != "shap":
            continue
        table = pd.read_csv(path)
        values = table[list(summary["mean_abs"])].to_numpy()
        total = values.sum(axis=1)
        gap = table["predicted"].to_numpy() - table["baseline"].to_numpy()
        np.testing.assert_allclose(total, gap, atol=1e-3)
        # Reading 1: the mean label replaces the reference prediction, the values stay.
        miss = table["predicted"].to_numpy() - (mean_label + total)
        assert np.ptp(miss) < 1e-3
        assert miss[0] == pytest.approx(table["baseline"].iloc[0] - mean_label, abs=1e-3)
        checked += 1
        try:
            saved = load_config(path.parents[2] / "preprocessing_config.yaml")
        except (FileNotFoundError, ValueError, TypeError):
            continue  # no settings file, or a setting renamed since; find_runs skips both
        if saved.digest() != digest:
            continue
        # Reading 2: a fourteenth of the gap added to every value adds up again,
        # but values of both signs change size in opposite directions.
        shift = (table["baseline"].iloc[0] - mean_label) / values.shape[1]
        shifted = values + shift
        np.testing.assert_allclose(mean_label + shifted.sum(axis=1), table["predicted"], atol=1e-3)
        before = np.argsort(-np.abs(values), axis=1, kind="stable")
        after = np.argsort(-np.abs(shifted), axis=1, kind="stable")
        reordered = int((before != after).any(axis=1).sum())
        assert reordered > 0, path
        if (path.parents[2].name, path.parent.name) == named:
            names = list(summary["mean_abs"])
            life = [names[i] for i in np.argsort(-np.abs(values).mean(axis=0))]
            life_shifted = [names[i] for i in np.argsort(-np.abs(shifted).mean(axis=0))]
            assert table["baseline"].iloc[0] == pytest.approx(91.885, abs=5e-4)
            assert shift == pytest.approx(0.803, abs=5e-4)
            assert (reordered, len(table)) == (236, 240)
            assert life[1:3] == ["W31", "T30"] and life_shifted[1:3] == ["T30", "W31"]
            assert life[11:13] == ["BPR", "Nf"] and life_shifted[11:13] == ["Nf", "BPR"]
            assert life[:1] + life[3:11] + life[13:] == life_shifted[:1] + life_shifted[3:11] + life_shifted[13:]
    if not checked:
        pytest.skip("no stored Shapley explanation of an FD001 run in reports/")


def test_example_a_different_stand_in_recomputes_and_reorders_the_values(loaded):
    """Two stand-ins for a straight line on real FD001 readings move each value differently and reorder them."""
    # The SHAP resource page's example; assumption A-24, corrected 2026-10-02.
    from thesis_gnn_rul_xai.data.sensors import symbol_to_column

    names = ["T24", "T30", "T50", "P30"]
    columns = [symbol_to_column(name) for name in names]
    train = apply_piecewise_rul(loaded.train, rul_max=125)
    readings = train[columns].to_numpy(dtype=np.float64)
    labels = train["rul"].to_numpy(dtype=np.float64)
    design = np.column_stack([readings, np.ones(len(readings))])
    w = np.linalg.lstsq(design, labels, rcond=None)[0][:4]
    x = loaded.test[loaded.test["unit"] == 5][columns].to_numpy(dtype=np.float64)[-1]

    # For a straight line, a sensor's Shapley value against a stand-in r is
    # its multiplier times (reading - stand-in).
    everyone = w * (x - readings.mean(axis=0))
    at_ceiling = w * (x - readings[labels == 125].mean(axis=0))
    np.testing.assert_allclose(everyone, [5.984611, 0.667847, -15.962813, -0.863789], atol=5e-6)
    np.testing.assert_allclose(at_ceiling, [1.883631, -3.150836, -24.780595, -6.969030], atol=5e-6)

    def by_size(values):
        return [names[i] for i in np.argsort(-np.abs(values))]

    assert by_size(everyone) == ["T50", "T24", "P30", "T30"]
    assert by_size(at_ceiling) == ["T50", "P30", "T30", "T24"]
    assert np.ptp(at_ceiling - everyone) > 1


def test_example_occlusion_can_equal_the_shapley_value_despite_interactions():
    """In explain/README.md's h(a, b, c) every input interacts, yet occlusion equals the Shapley value."""
    from itertools import permutations

    def h(a, b, c):
        return a + b + c + 2 * a * b + 2 * a * c + 2 * b * c - 3 * a * b * c

    def value(present):
        return h(*(1.0 if k in present else 0.0 for k in range(3)))

    # a's amounts by the group it finds on arrival: nobody, b alone, c alone, both.
    assert [value({0}) - value(set()), value({0, 1}) - value({1}),
            value({0, 2}) - value({2}), value({0, 1, 2}) - value({1, 2})] == [1, 3, 3, 2]
    for k in range(3):
        added = []
        for order in permutations(range(3)):
            before = set(order[: order.index(k)])
            added.append(value(before | {k}) - value(before))
        assert len(set(added)) > 1
        shapley = sum(added) / len(added)
        occlusion = value({0, 1, 2}) - value({0, 1, 2} - {k})
        assert shapley == occlusion == 2.0
    # So here the three occlusion values happen to add up to the whole change.
    assert 3 * 2.0 == value({0, 1, 2}) - value(set()) == 6.0


# --- scripts/baselines.py ---------------------------------------------


def test_example_the_three_summaries_an_ordinary_model_gets_for_one_window(raw_dir):
    """T50 of engine 1 over its cycles 1 to 30, smoothed and scaled under seed 42: last, average, slope."""
    # The window of the window-summary stage's example in models/temporal/README.md;
    # the baselines script gives its slope per cycle, where the stage multiplies by 29.
    import importlib.util
    import sys
    from dataclasses import replace
    from pathlib import Path

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    name = "baselines_script"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "scripts" / "baselines.py"
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    script = sys.modules[name]

    prepared = run_preprocessing(replace(PreprocessingConfig(), seed=42), data_dir=raw_dir)
    first = int(np.where(prepared.train.units == 1)[0][0])
    assert prepared.train.end_cycles[first] == 30
    sensor = list(prepared.sensor_symbols).index("T50")
    readings = prepared.train.features[first, sensor].astype(np.float64)

    # slope = sum(t x reading) / sum(t x t), with t = -14.5, -13.5, ..., +14.5
    steps = np.arange(30) - 14.5
    assert (steps * steps).sum() == 2247.5
    assert round(float((steps * readings).sum()), 6) == -27.887838
    slope = float((steps * readings).sum()) / 2247.5
    assert round(slope, 6) == -0.012408
    assert slope == pytest.approx(np.polyfit(np.arange(30), readings, 1)[0], abs=1e-12)

    # One row per window and 42 columns: 14 last readings, 14 averages, 14 slopes.
    summaries = script.model_inputs(prepared.train, "last+mean+slope")
    assert summaries.shape == (len(prepared.train), 42)
    row = summaries[first, [sensor, 14 + sensor, 28 + sensor]]
    assert [round(float(value), 4) for value in row[:2]] == [-1.2938, -1.0449]
    assert row[2] == pytest.approx(slope, abs=1e-12)


def test_example_a_paired_t_test_over_five_seeds_worked_by_hand():
    """The last-cycle network minus the recurrent one, seed by seed: t = 8.978, p = 0.00085, as the log records."""
    from scipy.stats import t as student_t
    from scipy.stats import ttest_rel

    from thesis_gnn_rul_xai.utils.paths import reports_dir

    names = {
        "last": "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z",
        "recurrent": "fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z",
    }
    paths = {key: reports_dir() / name / "results.csv" for key, name in names.items()}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("the two stored runs of 2026-09-14 are not in reports/")
    stored = {key: pd.read_csv(path).set_index("seed")["rmse"] for key, path in paths.items()}
    assert list(stored["last"].index) == list(stored["recurrent"].index) == [42, 43, 44, 45, 46]

    # The RMSE of each seed as printed in evaluation/README.md, to four places.
    last = np.array([round(float(v), 4) for v in stored["last"]])
    recurrent = np.array([round(float(v), 4) for v in stored["recurrent"]])
    assert last.tolist() == [16.6949, 16.1311, 16.8432, 15.9704, 16.8427]
    assert recurrent.tolist() == [12.4274, 13.8228, 13.2695, 13.4379, 13.5790]

    differences = last - recurrent  # the network's RMSE minus the other row's
    assert differences.round(4).tolist() == [4.2675, 2.3083, 3.5737, 2.5325, 3.2637]
    average = differences.sum() / 5
    assert round(average, 4) == 3.1891
    squares = (differences - average) ** 2
    assert squares.round(4).tolist() == [1.1629, 0.7759, 0.1479, 0.4312, 0.0056]
    spread = np.sqrt(squares.sum() / (5 - 1))
    assert round(spread, 4) == 0.7943
    standard_error = spread / np.sqrt(5)
    assert round(standard_error, 4) == 0.3552
    t = average / standard_error
    assert round(t, 3) == 8.978

    # Five differences whose true average is zero give a t between -2.776 and
    # +2.776 in 95 % of repetitions; 8.978 lies far outside.
    assert round(float(student_t.ppf(0.975, 5 - 1)), 3) == 2.776
    p = 2 * float(student_t.sf(abs(t), 5 - 1))
    assert round(p, 5) == 0.00085

    # The same from the unrounded scores, by scipy, which scripts/baselines.py uses.
    unrounded = ttest_rel(stored["last"].to_numpy(), stored["recurrent"].to_numpy())
    assert round(float(unrounded.statistic), 3) == 8.978
    assert round(float(unrounded.pvalue), 5) == 0.00085


# --- the audit of 2026-09-08: the every-sensor graph -----------------------


def test_example_the_every_sensor_graph_hands_every_sensor_the_same_average(raw_dir):
    """Joined to every other sensor, every sensor receives the plain average (assumption A-19)."""
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
    from thesis_gnn_rul_xai.graph.adjacency import build_fully_connected_adjacency

    prepared = run_preprocessing(PreprocessingConfig(), data_dir=raw_dir)
    readings = prepared.test.features[0, :, -1].astype(np.float64)  # engine 1, last cycle, 14 sensors
    assert prepared.test.units[0] == 1
    assert [round(float(v), 6) for v in readings[:3]] == [-0.427191, -1.224639, -1.10717]
    assert round(float(readings.mean()), 6) == -0.22801

    for method in ("row", "symmetric"):
        every = normalize_adjacency(build_fully_connected_adjacency()[0], method=method)
        every = every.astype(np.float64)
        np.testing.assert_allclose(every, np.full((14, 14), 1 / 14))
        assert np.linalg.matrix_rank(every) == 1
        np.testing.assert_allclose(every @ readings, np.full(14, readings.mean()))

        # The paper's graph, for contrast: nine distinct rows, eight independent (finding F-18).
        paper = normalize_adjacency(build_physics_adjacency()[0], method=method).astype(np.float64)
        assert len({tuple(np.round(row, 12)) for row in paper}) == 9
        assert np.linalg.matrix_rank(paper) == 8


# --- scripts/baselines.py: a tree model sees its inputs in single precision ---


def test_example_a_tree_model_rounds_its_inputs_to_single_precision(raw_dir):
    """Default settings, seed 42: rescaling the slope or re-rounding the average moves gradient boosting, not linear regression."""
    import importlib.util
    import sys
    from dataclasses import replace
    from pathlib import Path

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    name = "baselines_script"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "scripts" / "baselines.py"
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    script = sys.modules[name]

    # The settings and first seed of the stored default run of 2026-09-14.
    prepared = run_preprocessing(replace(PreprocessingConfig(), seed=42), data_dir=raw_dir)
    assert prepared.train.features.shape == (14459, 14, 30)
    assert prepared.train.features.dtype == np.float32
    assert (len(set(prepared.train.units)), len(set(prepared.validation.units))) == (80, 20)
    labels = prepared.train.targets.astype(np.float64)
    truth = prepared.test.targets.astype(np.float64)

    def predicted(model, train_inputs, test_inputs):
        model.fit(train_inputs, labels)
        return np.clip(model.predict(test_inputs), 0, 125)

    # The slope per cycle, and the same slope times 29 as the window-summary stage gives it.
    slope_train = script.model_inputs(prepared.train, "slope")
    slope_test = script.model_inputs(prepared.test, "slope")
    per_cycle = predicted(GradientBoostingRegressor(random_state=42), slope_train, slope_test)
    whole_window = predicted(GradientBoostingRegressor(random_state=42), 29 * slope_train, 29 * slope_test)
    assert round(rmse(truth, per_cycle), 3) == 13.320
    assert round(rmse(truth, whole_window), 3) == 13.238
    assert round(float(np.abs(per_cycle - whole_window).max()), 1) == 5.4
    # Linear regression takes a weight 29 times smaller instead, and predicts the same.
    linear_per_cycle = predicted(LinearRegression(), slope_train, slope_test)
    linear_whole_window = predicted(LinearRegression(), 29 * slope_train, 29 * slope_test)
    assert float(np.abs(linear_per_cycle - linear_whole_window).max()) < 1e-13

    # The average as the script takes it, in double precision, and as NumPy takes it in single.
    double = script.model_inputs(prepared.train, "mean")
    single = prepared.train.features.mean(axis=2)
    assert (double.dtype, single.dtype) == (np.float64, np.float32)
    differ = double.astype(np.float32) != single
    assert (int(differ.sum()), differ.size) == (106726, 202426)
    largest = float(np.abs(double.astype(np.float32).astype(np.float64) - single).max())
    assert largest == pytest.approx(7.2e-7, abs=5e-9)

    test_double = script.model_inputs(prepared.test, "mean")
    from_double = predicted(GradientBoostingRegressor(random_state=42), double, test_double)
    from_single = predicted(GradientBoostingRegressor(random_state=42), single, prepared.test.features.mean(axis=2))
    assert round(rmse(truth, from_double), 3) == 21.842
    assert round(rmse(truth, from_single), 3) == 21.689
    # Gradient boosting rounds the double-precision average to single precision itself:
    # rounding it by hand first changes no prediction.
    rounded_by_hand = predicted(
        GradientBoostingRegressor(random_state=42), double.astype(np.float32), test_double.astype(np.float32)
    )
    np.testing.assert_array_equal(rounded_by_hand, from_double)


# --- config.py: the six characters after "cfg" in a run's folder name ------


def test_example_the_configuration_digest_is_built_from_the_settings_without_the_notes():
    """The digest of config/fd001.yaml, by hand in three steps, and the default object giving the same."""
    import hashlib
    import json
    from pathlib import Path

    from thesis_gnn_rul_xai.config import (
        PreprocessingConfig,
        load_config,
        settings_without_notes,
    )

    config = load_config(Path(__file__).resolve().parents[1] / "config" / "fd001.yaml")
    # Step 1: the twelve saved fields with the prose notes left out, eleven settings.
    settings = settings_without_notes(config.to_dict())
    assert (len(config.to_dict()), len(settings)) == (12, 11)
    # Step 2: those settings as one line of text, the names in alphabetical order.
    text = json.dumps(settings, sort_keys=True, default=str)
    assert text == (
        '{"adjacency_normalization": "row", "clip_test_labels": true, "graph": "physics", '
        '"normalization": "zscore", "rul_max": 125, "seed": 42, "smoothing_alpha": 0.5, '
        '"subset": "FD001", "validation_fraction": 0.2, "window_size": 30, "window_stride": 1}'
    )
    # Step 3: the SHA-256 code of that text, and its first six characters.
    by_hand = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert by_hand.startswith("818b40380ec2c6a8")
    assert by_hand[:6] == config.digest() == "818b40"
    # The object built with no file has other notes and the same eleven settings.
    default = PreprocessingConfig()
    assert default.notes != config.notes
    assert default.digest() == "818b40"
    # config/fd001_uncapped.yaml differs in one setting, the cap on the test labels.
    uncapped = load_config(Path(__file__).resolve().parents[1] / "config" / "fd001_uncapped.yaml")
    assert [k for k in settings if settings[k] != settings_without_notes(uncapped.to_dict())[k]] == [
        "clip_test_labels"
    ]
    assert uncapped.digest() == "7c2c29"


# --- docs/assumptions/A-35.md: which window graph g of Table 12 is ---------


def test_example_reading_graph_g_as_another_window_moves_the_red_four_count_but_not_its_order():
    """The joins over Table 12's 12 steps on stored engine 5, at every shift of graph g's window."""
    import json

    from thesis_gnn_rul_xai.evaluation.published import (
        SHAP_TABLE_12,
        SHAP_TABLE_12_GRAPHS,
    )
    from thesis_gnn_rul_xai.utils.paths import reports_dir

    stored = {
        "last_cycle": ("fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z",
                       "shap_engine_unit5_seed42_f386291-dirty_20260914T161425Z"),
        "lstm": ("fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z",
                 "shap_engine_unit5_seed42_f386291-dirty_20260914T135414Z"),
    }
    folders = {stage: reports_dir() / run / "explanations" / name for stage, (run, name) in stored.items()}
    if not all((folder / "values.csv").is_file() for folder in folders.values()):
        pytest.skip("The two stored Shapley explanations of engine 5 made on 2026-09-14 are not in reports/.")
    sensors = list(SHAP_TABLE_12)

    def red_four(values):
        order = sorted(range(len(sensors)), key=lambda i: -abs(values[i]))
        return {sensors[i] for i in order[:4]}

    # The 12 steps inside graphs 1-5, 78-82 and 187-191, as pairs of Table 12's fifteen columns.
    steps = [(k, k + 1) for start in (0, 5, 10) for k in range(start, start + 4)]
    assert len(steps) == 12

    def joins(rows):
        return sum(len(red_four(rows[after]) - red_four(rows[before])) for before, after in steps)

    paper = [[SHAP_TABLE_12[s][column] for s in sensors] for column in range(15)]
    assert joins(paper) == 1

    counted = {}
    for stage, folder in folders.items():
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        assert (summary["unit"], summary["seed"], summary["every"]) == (5, 42, 1)
        table = pd.read_csv(folder / "values.csv").sort_values("end_cycle")
        # Every window of engine 5, one cycle apart, the first ending at cycle 30.
        assert table["end_cycle"].tolist() == list(range(30, 270))
        values = table[sensors].to_numpy()
        # The reading used: graph g is row g - 1, the window ending at cycle g + 29.
        assert table["end_cycle"].iloc[SHAP_TABLE_12_GRAPHS[-1] - 1] == 220
        # Shift k puts graph g at the window ending at cycle g + 29 + k; the
        # last shift puts graph 191 at the engine's last window, cycle 269.
        shifts = range(len(values) - SHAP_TABLE_12_GRAPHS[-1] + 1)
        counted[stage] = np.array([joins(values[[g - 1 + k for g in SHAP_TABLE_12_GRAPHS]]) for k in shifts])
    assert {stage: len(c) for stage, c in counted.items()} == {"last_cycle": 50, "lstm": 50}
    # Our reading, the reading from the engine's failure, and the fewest and most over every shift.
    assert {stage: (c[0], c[-1], c.min(), c.max()) for stage, c in counted.items()} == {
        "last_cycle": (17, 9, 9, 19), "lstm": (2, 2, 0, 7)}
    # At every shift the default network has more joins than the recurrent one and than Table 12.
    assert (counted["last_cycle"] > counted["lstm"]).all() and counted["last_cycle"].min() > joins(paper)
    # The recurrent network has one join or none at 15 of the 50 shifts on this seed.
    assert int((counted["lstm"] <= joins(paper)).sum()) == 15


# --- config.py: a setting written as 0 or as 0.0 gives one digest ------------


def test_example_a_whole_number_where_a_decimal_is_declared_is_held_as_the_decimal():
    """validation_fraction 0 is held as 0.0, so its JSON text and its digest are those of 0.0."""
    import json

    from thesis_gnn_rul_xai.config import PreprocessingConfig, settings_without_notes

    # Python counts the two as equal, which is how find_runs compares settings,
    # but as JSON text, which is what the digest reads, they are two lines.
    assert 0 == 0.0
    assert (json.dumps(0), json.dumps(0.0)) == ("0", "0.0")
    written = PreprocessingConfig(validation_fraction=0)
    assert type(written.validation_fraction) is float
    text = json.dumps(settings_without_notes(written.to_dict()), sort_keys=True, default=str)
    assert '"validation_fraction": 0.0' in text
    assert written.digest() == PreprocessingConfig(validation_fraction=0.0).digest() == "2c9b5b"
    # A whole-number field goes the other way: 125.0 is held as 125.
    assert type(PreprocessingConfig(rul_max=125.0).rul_max) is int


# --- evaluation/metrics.py: ASUE when no engine is predicted early (A-39) ----


def test_example_asue_is_undefined_for_a_model_that_is_never_early(raw_dir):
    """Predicting the ceiling for every FD001 test engine is late on all 100, so ASUE has nothing to average."""
    labels = np.minimum(np.loadtxt(raw_dir / "RUL_FD001.txt"), 125)   # the 100 answers, capped
    assert labels.shape == (100,) and labels.max() == 125
    with pytest.warns(RuntimeWarning, match="no engine was predicted early"):
        scored = evaluate(labels, np.full(100, 125.0))
    # No capped answer is above 125, so no prediction is below its answer:
    # Equation 24's N_under is 0 and its sum is 0.
    assert (scored["n_early"], scored["n_late"]) == (0, 100)
    assert np.isnan(scored["asue"])
    # Late on every engine, by 50.55 cycles on average: the mistake the NASA
    # score punishes hardest. The 0.0 returned until 2026-10-03 called this
    # the best ASUE there is.
    assert round(scored["mean_error"], 2) == 50.55
    assert round(scored["score"], 1) == 1502460.8
    # One cycle lower, the 11 engines whose answers were capped from above 125
    # are each early by exactly one cycle, so ASUE is (1 + ... + 1) / 11.
    lower = evaluate(labels, np.full(100, 124.0))
    assert (lower["n_early"], lower["asue"]) == (11, 1.0)
    assert (np.flatnonzero(labels > 124.0) + 1).tolist() == [25, 39, 47, 55, 65, 73, 74, 83, 89, 95, 96]


# --- graph/controls.py: graphs with Table 3's 41 connections but not its choices (A-41) ---


def test_example_one_swap_on_table_3_moves_two_connections_and_keeps_four_counts():
    """P15-Nf and T50-W32 become P15-W32 and T50-Nf; all four sensors still have 4 connections each."""
    from thesis_gnn_rul_xai.graph.controls import (
        build_random_adjacency,
        build_rewired_adjacency,
        swap_ends,
    )

    table3, symbols = build_physics_adjacency()
    at = {symbol: i for i, symbol in enumerate(symbols)}
    count = dict(zip(symbols, table3.sum(axis=1).astype(int).tolist(), strict=True))
    assert [count[s] for s in ("P15", "Nf", "T50", "W32")] == [4, 4, 4, 4]
    first, second = (at["P15"], at["Nf"]), (at["T50"], at["W32"])
    assert (first, second) == ((3, 5), (2, 13))
    assert table3[first] == table3[second] == 1                  # both are in Table 3
    new_first, new_second = swap_ends(first, second)
    assert (new_first, new_second) == ((3, 13), (2, 5))          # P15-W32 and T50-Nf
    assert table3[new_first] == table3[new_second] == 0          # neither is, yet
    swapped = table3.copy()
    for (i, j), value in ((first, 0), (second, 0), (new_first, 1), (new_second, 1)):
        swapped[i, j] = swapped[j, i] = value
    # P15 lost Nf and gained W32, Nf lost P15 and gained T50, and so on: 4 each.
    np.testing.assert_array_equal(swapped.sum(axis=1), table3.sum(axis=1))
    assert int(swapped.sum() // 2) == 41

    # A swap that would join one pair twice is refused: T24-T30 with T50-W32
    # would give T24-W32 and T30-T50, and T30-T50 is already in Table 3.
    assert swap_ends((at["T24"], at["T30"]), (at["T50"], at["W32"])) == ((0, 13), (1, 2))
    assert table3[at["T30"], at["T50"]] == 1

    # Seed 42's 4,000 proposals end with every count as Table 3 has it and 19
    # of the 41 connections the engineer drew; Nf-NRf, a speed and the
    # corrected speed computed from it, is not among them.
    rewired, _ = build_rewired_adjacency(42)
    assert rewired.sum(axis=1).astype(int).tolist() == [7, 8, 4, 4, 7, 4, 8, 7, 6, 4, 4, 7, 8, 4]
    assert int((rewired * table3).sum() // 2) == 19
    assert table3[at["Nf"], at["NRf"]] == 1 and rewired[at["Nf"], at["NRf"]] == 0
    # The random graph keeps only the count: seed 42 shares 19 connections too,
    # but gives P30 11 where Table 3 gives it 7.
    drawn, _ = build_random_adjacency(42)
    assert int(drawn.sum() // 2) == 41 and int((drawn * table3).sum() // 2) == 19
    assert (int(drawn[at["P30"]].sum()), count["P30"]) == (11, 7)


def test_example_a_random_41_connection_graph_on_14_sensors_is_split_once_in_386_draws():
    """Counting every such graph: 0.259 % leave some sensor unable to reach another, nearly all by leaving one alone."""
    from functools import cache
    from math import comb

    @cache
    def in_one_piece(n: int, m: int) -> int:
        """Graphs on n named sensors with m connections in which every sensor reaches every other."""
        if n == 1:
            return int(m == 0)
        total = comb(comb(n, 2), m)
        # Take away every graph in which sensor 1's piece holds only k sensors.
        for k in range(1, n):
            for j in range(m + 1):
                total -= comb(n - 1, k - 1) * in_one_piece(k, j) * comb(comb(n - k, 2), m - j)
        return total

    # Four sensors and three connections, by hand: C(6, 3) = 20 graphs; the 4
    # triangles leave one sensor alone, the other 16 reach everything.
    assert (comb(6, 3), in_one_piece(4, 3)) == (20, 16)
    assert [in_one_piece(4, m) for m in range(7)] == [0, 0, 0, 16, 15, 6, 1]

    every = comb(comb(14, 2), 41)                       # 41 of the 91 possible pairs
    split = 1 - in_one_piece(14, 41) / every
    assert round(split, 6) == 0.002592 and round(1 / split) == 386
    # Some sensor joined to nothing, by counting in and out over which are alone.
    alone = sum((-1) ** (k + 1) * comb(14, k) * comb(comb(14 - k, 2), 41) for k in range(1, 15)) / every
    assert round(alone, 6) == 0.00259 and alone / split > 0.999


# --- Which way the three graphs' test errors lean, against the label and against
# the life truly left (added 2026-10-03). The log's comparison of the graphs
# (assumption A-27) and step 5 of notebooks/graph/04_comparison_fd001.ipynb quote
# these figures; the notebook does not compute the uncapped ones itself.

LEAN_RUNS = {
    "Table 3": "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z",
    "correlation": "fd001_correlation_row_cap_5seed_varsplit_cfg7fdfce_f386291-dirty_20260914T150213Z",
    "every sensor": "fd001_complete_row_cap_5seed_varsplit_cfgf21547_f386291-dirty_20260914T150527Z",
}


def test_example_the_lean_of_the_errors_against_the_capped_label_and_against_the_life_truly_left(raw_dir):
    """Every seed's average error is 1.070 cycles lower against NASA's answer than against the label capped at 125."""
    import json
    from pathlib import Path

    from scipy import stats

    from thesis_gnn_rul_xai.data.loading import read_rul_file
    from thesis_gnn_rul_xai.utils.paths import project_root, reports_dir

    paths = {graph: reports_dir() / name for graph, name in LEAN_RUNS.items()}
    if not all((path / "seed_46" / "test_predictions.csv").is_file() for path in paths.values()):
        pytest.skip("the three stored runs of 2026-09-14 are not in reports/")

    # NASA's answer, a pandas Series indexed by test engine 1 to 100, and the
    # label every score is computed against: the same, capped at 125.
    life = read_rul_file(raw_dir / "RUL_FD001.txt")
    label = np.minimum(life, 125)
    # 11 engines have more than 125 cycles left, 107 cycles above the cap between
    # them, so the average label is 107 / 100 = 1.070 below the average life.
    assert int((life > 125).sum()) == 11 and int((life - label).sum()) == 107

    by_label, by_life, highest = {}, {}, {}
    for graph, path in paths.items():
        by_label[graph], by_life[graph], highest[graph] = [], [], 0.0
        for seed in range(42, 47):
            saved = pd.read_csv(path / f"seed_{seed}" / "test_predictions.csv").set_index("unit")
            truly_left = life.loc[saved.index].to_numpy()
            np.testing.assert_array_equal(saved["rul"].to_numpy(), np.minimum(truly_left, 125))
            late_by_label = saved["predicted"] - saved["rul"]
            late_by_life = saved["predicted"] - truly_left
            # No prediction reaches 125, so a prediction above its label is never on a capped
            # engine: the late engines and the largest late error are the same either way.
            assert int((late_by_label > 0).sum()) == int((late_by_life > 0).sum())
            assert late_by_label.max() == pytest.approx(late_by_life.max(), abs=1e-9)
            by_label[graph].append(float(late_by_label.mean()))
            by_life[graph].append(float(late_by_life.mean()))
            highest[graph] = max(highest[graph], float(saved["predicted"].max()))
        by_label[graph], by_life[graph] = np.array(by_label[graph]), np.array(by_life[graph])
        np.testing.assert_allclose(by_label[graph] - by_life[graph], 1.07, atol=1e-9)

    def rounded(values):
        return [round(float(v), 3) for v in values]

    # Seed 42 of Table 3, the first of the five: 1.158 above the label, 0.088 above the life.
    assert rounded(by_label["Table 3"]) == [1.158, 1.543, 2.328, 1.036, -1.763]
    assert rounded(by_life["Table 3"]) == [0.088, 0.473, 1.258, -0.034, -2.833]
    assert rounded(by_label["correlation"]) == [2.661, 3.2, 4.252, 2.289, 0.655]
    assert rounded(by_life["correlation"]) == [1.591, 2.13, 3.182, 1.219, -0.415]
    summary = {graph: (round(float(by_life[graph].mean()), 3), round(float(by_life[graph].std(ddof=1)), 3))
               for graph in LEAN_RUNS}
    assert summary == {"Table 3": (-0.21, 1.551), "correlation": (1.541, 1.32), "every sensor": (1.801, 1.615)}
    # Against the label the correlation graph is above it on every seed; against the life, not on seed 46.
    assert (by_label["correlation"] > 0).all() and by_life["correlation"][-1] < 0
    assert round(float(stats.ttest_1samp(by_life["Table 3"], 0).pvalue), 2) == 0.78
    assert round(float(stats.ttest_1samp(by_life["correlation"], 0).pvalue), 3) == 0.059
    assert round(max(highest["Table 3"], highest["correlation"]), 3) == 123.356
    assert round(max(highest.values()), 3) == 123.754

    # The two places that quote these figures, in their own minus sign.
    def shown(value):
        return f"{value:.3f}".replace("-", "\N{MINUS SIGN}")

    log = (project_root() / "docs" / "assumptions" / "A-27.md").read_text(encoding="utf-8")
    note = log[log.index("**They do make different errors.**"):log.index("**What this says about the paper's claim.**")]
    for figure in (f"{shown(-0.210)} ± 1.551", "1.541 ± 1.320", shown(-0.415), "123.356", "1.070"):
        assert figure in note, figure
    notebook = json.loads(Path(project_root(), "notebooks", "graph", "04_comparison_fd001.ipynb")
                          .read_text(encoding="utf-8"))
    step5 = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "3c91166f")
    for figure in (f"{shown(-0.210)} ± 1.551", "1.541 ± 1.320", shown(-0.415), "1.801 ± 1.615", "1.070"):
        assert figure in step5, figure
