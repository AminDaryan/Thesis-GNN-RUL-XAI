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
    for key in playground._READ_MORE:
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
        assert field["key"] in playground._READ_MORE, field["key"]


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
    # Two settings whose options come from different places, and two nothing sets.
    assert untagged_rows == {"graph", "temporal", "seed", "device"}
    for key in ("graph", "temporal"):
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
    for key in ("without_lstm", "without_physics_prior"):
        moved = {k: v for k, v in variants[key].items() if full[k] != v}
        assert len(moved) == 1, key
        taken_out.update(moved)
    assert removed == taken_out == {"temporal": "last_cycle", "graph": "complete"}
    # A model without graph layers is not built, so that box has no cross.
    assert variants["without_gcn"] is None
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
        flat += step["steps"] if step.get("kind") in ("group", "branch") else [step]
    return [step for step in flat if step.get("kind") != "edge"]


def _step(answer: dict, name: str) -> dict:
    return next(step for step in _steps(answer) if step["name"] == name)


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
    assert "Two sets of windows" in names and "Three sets of windows" not in names
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
    assert _step(answer, "Training engines")["shape"] == f"({playground._count(len(loaded.train))}, 26)"
    assert _step(answer, "Test engines")["shape"] == f"({playground._count(len(loaded.test))}, 26)"


def test_the_preprocessing_box_counts_the_windows_the_pipeline_makes(raw_dir):
    from thesis_gnn_rul_xai.config import PreprocessingConfig
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    settings = playground.validate_settings({"window_size": 25})
    prepared = run_preprocessing(PreprocessingConfig(**playground.preprocessing_settings(settings)))
    step = _step(playground.stage_detail("preprocessing", settings), "Three sets of windows")
    count = playground._count
    assert step["parts"][0]["text"] == (f"{count(len(prepared.train))} training · "
                                        f"{count(len(prepared.validation))} held back · {count(len(prepared.test))} test")


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
    assert best["text"] == ("The paper never describes restoring the best pass; on FD001 it moved "
                            "what the LSTM is worth from 0.77 to 3.19 cycles.")
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
    assert set(answer["dataset"]) == {"table5", "counted"}
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


@pytest.mark.parametrize("name", ["C:", "C:Windows", "../secret", "..", "", ".hidden", "sub/dir"])
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
