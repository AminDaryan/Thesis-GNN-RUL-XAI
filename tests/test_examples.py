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
    stored = next((p for p in sorted((root / "reports").glob("*/explanations/*/summary.json"))
                   if json.loads(p.read_text(encoding="utf-8")).get("method") == "walks"), None)
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
    stored = next((p for p in sorted((root / "reports").glob("*/explanations/*/summary.json"))
                   if (s := json.loads(p.read_text(encoding="utf-8"))).get("method") == "walks"
                   and s.get("scope") == scope), None)
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


def _check_design_tutorial(filename, monkeypatch, minimum_tables=10):
    """Execute the page in order and compare its named samples with every printed cell."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    document = (root / "docs" / "learn" / filename).read_text(encoding="utf-8")
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
    """Only the two pessimistic engines count: (10 + 20) / 2."""
    assert asue([100, 50, 30], [90, 60, 10]) == 15.0
    assert asue([100], [110]) == 0.0


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

