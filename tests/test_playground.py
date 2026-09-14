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


# -- settings ------------------------------------------------------------------


def test_every_default_passes_its_own_check():
    settings = playground.validate_settings({})
    assert settings == {f["key"]: f["default"] for f in playground.FIELDS}


def test_every_setting_says_where_its_default_comes_from():
    kinds = {"paper", "assumption", "project", "extension"}
    for field in playground.FIELDS:
        assert field["source"]["kind"] in kinds, field["key"]
        if field["source"]["kind"] in ("paper", "assumption"):
            assert field["source"]["ref"], field["key"]
        for choice in field.get("choices", []):
            assert choice["source"]["kind"] in kinds, (field["key"], choice["value"])


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


def test_fewer_than_five_seeds_is_marked_exploratory():
    notes = playground.caveats(playground.validate_settings({"seeds": 1}))
    assert any(note["ref"] == "rule 3" for note in notes)
    assert not any(
        note["ref"] == "rule 3" for note in playground.caveats(playground.validate_settings({}))
    )


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


def test_the_site_explains_a_refused_setting(site: str):
    status, answer = _post(site + "/api/playground/preview", {"seeds": 0}, origin=site)
    assert status == 400
    assert "Seeds" in answer["error"]


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
