"""The random and rewired sensor graphs that stand in for the paper's graph without its knowledge.

src/thesis_gnn_rul_xai/graph/README.md explains both graphs and why they are
drawn the way they are; docs/assumptions/A-41.md records the choices.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from thesis_gnn_rul_xai.config import SENSOR_GRAPHS, PreprocessingConfig
from thesis_gnn_rul_xai.data.sensors import PAPER_SENSOR_SYMBOLS
from thesis_gnn_rul_xai.graph.adjacency import (
    adjacency_report,
    build_physics_adjacency,
    connections_of,
    normalize_adjacency,
)
from thesis_gnn_rul_xai.graph.controls import (
    build_random_adjacency,
    build_rewired_adjacency,
    graph_generator,
    swap_ends,
)
from thesis_gnn_rul_xai.utils.paths import reports_dir

ROOT = Path(__file__).resolve().parents[1]
PHYSICS, SYMBOLS = build_physics_adjacency()
# The five seeds a five-seed run trains, and a wider range for the properties
# that must hold on every draw.
RUN_SEEDS = range(42, 47)
MANY_SEEDS = range(200)
BUILDERS = {"random": build_random_adjacency, "rewired": build_rewired_adjacency}


def _shared(a: np.ndarray, b: np.ndarray) -> int:
    """How many connections two graphs have in common."""
    return int((np.asarray(a) * np.asarray(b)).sum() // 2)


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_every_draw_has_41_connections_both_ways_and_no_sensor_joined_to_itself(kind):
    for seed in MANY_SEEDS:
        matrix, symbols = BUILDERS[kind](seed)
        assert matrix.shape == (14, 14) and matrix.dtype == np.float32
        assert set(np.unique(matrix)) <= {0.0, 1.0}
        assert np.array_equal(matrix, matrix.T), seed
        assert not np.diag(matrix).any(), seed
        assert int(matrix.sum() // 2) == 41, seed
        assert symbols == PAPER_SENSOR_SYMBOLS


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_every_draw_joins_all_fourteen_sensors_into_one_piece(kind):
    for seed in MANY_SEEDS:
        assert adjacency_report(BUILDERS[kind](seed)[0], SYMBOLS)["is_connected"], seed


def test_the_rewiring_keeps_every_sensors_number_of_connections():
    degrees = PHYSICS.sum(axis=1)
    assert degrees.astype(int).tolist() == [7, 8, 4, 4, 7, 4, 8, 7, 6, 4, 4, 7, 8, 4]
    for seed in MANY_SEEDS:
        np.testing.assert_array_equal(build_rewired_adjacency(seed)[0].sum(axis=1), degrees)


def test_the_random_graph_does_not_keep_the_degrees():
    """Only the count of connections is held; the five run seeds give five different degree lists."""
    lists = {tuple(sorted(build_random_adjacency(seed)[0].sum(axis=1).astype(int))) for seed in RUN_SEEDS}
    assert len(lists) == 5
    assert tuple(sorted(PHYSICS.sum(axis=1).astype(int))) not in lists


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_the_same_seed_gives_the_same_graph_and_another_seed_another(kind):
    build = BUILDERS[kind]
    for seed in RUN_SEEDS:
        np.testing.assert_array_equal(build(seed)[0], build(seed)[0])
    drawn = [build(seed)[0] for seed in RUN_SEEDS]
    assert len({matrix.tobytes() for matrix in drawn}) == len(drawn)


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_neither_control_is_the_papers_graph_on_any_run_seed(kind):
    """Measured 2026-10-03: they share 16 to 25 of the 41 connections with Table 3."""
    shared = [_shared(BUILDERS[kind](seed)[0], PHYSICS) for seed in RUN_SEEDS]
    assert shared == {"random": [19, 20, 16, 20, 20], "rewired": [19, 21, 21, 25, 21]}[kind]


def test_the_graph_draws_numbers_of_its_own_not_the_splits():
    """The split starts NumPy's generator from the seed alone; the graph starts it from (seed, 1)."""
    split = np.random.default_rng(42).random(5)
    graph = graph_generator(42).random(5)
    assert not np.allclose(split, graph)
    np.testing.assert_array_equal(graph, np.random.default_rng([42, 1]).random(5))


def test_a_draw_that_leaves_a_sensor_alone_is_drawn_again():
    """Seed 254 is the first of 0, 1, 2, ... whose first draw splits the sensors: W31 is joined to nothing."""
    first, _ = build_random_adjacency(254, connected=False)
    assert [SYMBOLS[i] for i in np.flatnonzero(first.sum(axis=1) == 0)] == ["W31"]
    assert not adjacency_report(first, SYMBOLS)["is_connected"]
    for seed in range(254):
        assert adjacency_report(build_random_adjacency(seed, connected=False)[0], SYMBOLS)["is_connected"]
    kept, _ = build_random_adjacency(254)
    assert adjacency_report(kept, SYMBOLS)["is_connected"]
    assert not np.array_equal(first, kept)


def test_swap_ends_exchanges_the_second_ends_and_lists_each_pair_lower_first():
    assert swap_ends((3, 5), (2, 13)) == ((3, 13), (2, 5))
    assert swap_ends((0, 1), (2, 13)) == ((0, 13), (1, 2))


def test_the_builders_refuse_what_they_cannot_draw():
    with pytest.raises(ValueError, match="cannot join 14 sensors"):
        build_random_adjacency(0, n_edges=12)
    with pytest.raises(ValueError, match="n_edges must lie"):
        build_random_adjacency(0, n_edges=92)
    with pytest.raises(ValueError, match="symmetric"):
        build_rewired_adjacency(0, adjacency=np.triu(PHYSICS))
    with pytest.raises(ValueError, match="proposals"):
        build_rewired_adjacency(0, proposals=0)


def test_both_controls_can_be_rescaled_and_lift_the_papers_rank_of_8():
    """Table 3 gives 8 independent rows after rescaling (finding F-18); every control drawn here gives 14."""
    assert np.linalg.matrix_rank(normalize_adjacency(PHYSICS, method="row").astype(np.float64)) == 8
    for build in BUILDERS.values():
        for seed in RUN_SEEDS:
            table = normalize_adjacency(build(seed)[0], method="row").astype(np.float64)
            np.testing.assert_allclose(table.sum(axis=1), 1.0, rtol=1e-6)
            assert np.linalg.matrix_rank(table) == 14


def test_the_configuration_offers_both_graphs_and_keeps_its_eleven_settings():
    assert {"random", "rewired"} <= set(SENSOR_GRAPHS)
    for kind in ("random", "rewired"):
        assert PreprocessingConfig(graph=kind).graph == kind
    with pytest.raises(ValueError, match="graph must be one of"):
        PreprocessingConfig(graph="erdos")


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_the_pipeline_draws_the_graph_from_the_seed_it_prepares_under(kind, raw_dir):
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    for seed in (7, 43):
        prepared = run_preprocessing(
            PreprocessingConfig(graph=kind, window_size=20, validation_fraction=0.25, seed=seed),
            data_dir=raw_dir,
        )
        np.testing.assert_array_equal(prepared.adjacency_raw, BUILDERS[kind](seed)[0])
        assert prepared.diagnostics["graph_kind"] == kind
        assert prepared.diagnostics["graph_seed"] == seed
        assert prepared.diagnostics["graph"]["n_edges"] == 41
        assert prepared.diagnostics["graph"]["is_connected"] is True


def test_the_pipeline_records_no_graph_seed_for_a_graph_not_drawn_at_random(raw_dir):
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    prepared = run_preprocessing(PreprocessingConfig(window_size=20, seed=7), data_dir=raw_dir)
    assert prepared.diagnostics["graph_seed"] is None


def _train_script():
    """scripts/train.py loaded as a module, without running it."""
    import importlib.util

    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("_train_script_graph", ROOT / "scripts" / "train.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_the_training_script_writes_each_seeds_graph_by_sensor_name(raw_dir):
    from thesis_gnn_rul_xai.data.pipeline import run_preprocessing

    script = _train_script()
    prepared = run_preprocessing(
        PreprocessingConfig(graph="rewired", window_size=20, seed=42), data_dir=raw_dir
    )
    record = script.graph_record(prepared)
    assert record["graph"] == "rewired" and record["drawn_from_seed"] == 42
    assert len(record["connections"]) == 41
    index = {symbol: i for i, symbol in enumerate(SYMBOLS)}
    pairs = [(index[a], index[b]) for a, b in record["connections"]]
    assert pairs == connections_of(build_rewired_adjacency(42)[0])
    json.dumps(record)  # graph.json is written from this


def test_the_training_script_refuses_a_fixed_random_graph_with_nothing_held_back(tmp_path):
    """With nothing held back every seed is prepared under its own seed, so the one graph asked for cannot be given."""
    pytest.importorskip("torch")
    config = tmp_path / "nothing_held_back.yaml"
    PreprocessingConfig(validation_fraction=0.0).save(config)
    finished = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(config),
         "--graph", "random", "--fixed-split", "--output", str(tmp_path / "never")],
        capture_output=True, text=True, timeout=300,
    )
    assert finished.returncode == 2
    assert "needs engines held back" in finished.stderr
    assert not (tmp_path / "never").exists()


def _stored_graph_runs() -> list[Path]:
    """Every stored run, at any depth below reports/, whose seed folders record their graph by name."""
    root = reports_dir()
    found = {record.parent.parent for record in root.rglob("seed_*/graph.json")} if root.is_dir() else set()
    return sorted(path for path in found if (path / "summary.json").is_file())


# A sensitivity study keeps its runs inside its own folder, so the search goes
# below the top level of reports/.
STORED = _stored_graph_runs()


@pytest.mark.skipif(not STORED, reason="no stored run in reports/ records its graph by name yet")
@pytest.mark.parametrize("path", STORED, ids=[path.relative_to(reports_dir()).as_posix() for path in STORED])
def test_a_stored_runs_graph_is_the_one_its_network_holds_and_its_seed_rebuilds(path, raw_dir):
    """graph.json, the adjacency saved inside model.pt, and the graph a reload rebuilds must agree."""
    torch = pytest.importorskip("torch")
    from thesis_gnn_rul_xai.training import load_seed, read_run

    run = read_run(path)
    seeds = run.summary.get("graph_seeds")
    index = {symbol: i for i, symbol in enumerate(SYMBOLS)}
    for position, seed in enumerate(run.seeds):
        folder = path / f"seed_{seed}"
        record = json.loads((folder / "graph.json").read_text(encoding="utf-8"))
        assert record["graph"] == run.preprocessing.graph
        recorded = sorted((index[a], index[b]) for a, b in record["connections"])
        held = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)["adjacency"].numpy()
        off_diagonal = held.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        assert recorded == connections_of((off_diagonal != 0).astype(np.float32))
        # The graph a reload rebuilds, by the rule training/stored.py follows,
        # must be the one this seed was trained on. Loading the weights puts
        # the trained graph back into the network whatever was rebuilt, so the
        # network's predictions would not show a difference.
        _, prepared = load_seed(run, seed, data_dir=raw_dir)
        assert connections_of(prepared.adjacency_raw) == recorded
        assert prepared.diagnostics["graph_seed"] == record["drawn_from_seed"]
        if run.preprocessing.graph in ("random", "rewired"):
            assert seeds[position] == record["drawn_from_seed"]


# --- the documentation site, when a stored run used one of these graphs -------


@pytest.mark.parametrize("kind", ["random", "rewired"])
@pytest.mark.parametrize("temporal", ["last_cycle", "lstm"])
def test_the_playground_sets_no_row_of_the_paper_beside_these_graphs(kind, temporal):
    from thesis_gnn_rul_xai.docsite import playground

    row, reason = playground.paper_row_for("FD001", temporal, kind)
    assert row is None
    drawn = {"random": "a graph drawn at random", "rewired": "a rewiring of its graph"}[kind]
    assert reason == (f"The paper never describes {drawn}, and never says what its w/o Physics "
                      "prior row used instead; only every sensor joined is set beside that row, "
                      "so no row describes this run.")


def test_the_playground_says_whether_each_seed_drew_its_own_graph():
    from thesis_gnn_rul_xai.docsite import playground

    def note(**settings):
        return [c["text"] for c in playground.caveats({"seeds": 5, **settings}) if c["ref"] == "A-41"]

    assert note(graph="random") == [
        "The graph is drawn at random from each seed, so every seed is given a different graph."]
    # One graph serves every seed only when the same engines are held back for
    # all of them; with none held back the training script refuses the run, and
    # the caution says so (corrected 2026-10-03: it said "one seed" here too).
    assert note(graph="rewired", fixed_split=True, validation_fraction=0.2) == [
        "The graph is drawn at random from one seed and serves every seed."]
    assert note(graph="rewired", fixed_split=True, validation_fraction=0.0) == [
        "Same held-back engines for every seed, with a graph drawn at random, needs engines held back: "
        "with none, every seed is prepared under its own seed and would be given a graph of its own, "
        "so the training script refuses these settings."]
    assert note(graph="physics") == []


def test_each_rewiring_reports_how_many_of_its_proposals_were_accepted(caplog):
    """Measured 2026-10-03 on the five run seeds: 858 to 930 of the 4,000; each replaces two of the 41 connections."""
    import logging

    with caplog.at_level(logging.INFO, logger="thesis_gnn_rul_xai.graph.controls"):
        for seed in RUN_SEEDS:
            build_rewired_adjacency(seed)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Rewired")]
    assert [int(line.split("; ")[1].split()[0]) for line in lines] == [866, 930, 858, 868, 887]
    assert all("in 1 round(s) of 4000 proposal(s)" in line for line in lines)


def test_a_swap_and_its_undoing_are_each_made_by_two_of_the_3362_choices():
    """graph/README.md: why a refused swap still counts. P15-Nf with T50-W32, on Table 3, and back."""

    def results(edges):
        """The graph each of the 41 x 41 x 2 choices of one step leads to, refused ones staying put."""
        present, out = set(edges), []
        for i in range(len(edges)):
            for j in range(len(edges)):
                for flip in (0, 1):
                    second = edges[j][::-1] if flip else edges[j]
                    new = swap_ends(edges[i], second)
                    if i == j or any(a == b for a, b in new) or set(new) & present:
                        out.append(frozenset(edges))
                    else:
                        out.append(frozenset(present - {edges[i], edges[j]} | set(new)))
        return out

    table3 = connections_of(PHYSICS)
    at = {symbol: i for i, symbol in enumerate(SYMBOLS)}
    before = frozenset(table3)
    after = frozenset(before - {(at["P15"], at["Nf"]), (at["T50"], at["W32"])}
                      | {(at["P15"], at["W32"]), (at["T50"], at["Nf"])})
    forward, backward = results(table3), results(sorted(after))
    assert len(forward) == len(backward) == 41 * 41 * 2 == 3362
    assert forward.count(after) == backward.count(before) == 2


def test_each_random_graph_of_the_five_run_seeds_is_in_one_piece_at_its_first_draw(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="thesis_gnn_rul_xai.graph.controls"):
        for seed in RUN_SEEDS:
            build_random_adjacency(seed)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Drew")]
    assert len(lines) == 5 and all(line.endswith("in 1 draw(s).") for line in lines)


def test_setting_a_random_graph_run_up_again_keeps_the_forms_default_and_says_so(tmp_path):
    """The form offers no graph drawn at random, so a stored run that used one cannot put it into the form."""
    import re
    import shutil

    from thesis_gnn_rul_xai.docsite import playground

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    source = (ROOT / "src" / "thesis_gnn_rul_xai" / "docsite" / "assets" / "playground.js").read_text(encoding="utf-8")
    function = re.search(r"  function settingsFromRun\(d\) \{.*?\n  \}\n", source, re.S).group(0)
    script = tmp_path / "again.js"
    script.write_text(
        "var toasts = []; var T = {toast: function (m) { toasts.push(m); }};\n"
        "function copy(x) { return JSON.parse(JSON.stringify(x)); }\n"
        f"var S = {{options: {json.dumps(playground.options(), default=str)}}};\n"
        "S.fields = {}; S.options.fields.forEach(function (f) { S.fields[f.key] = f; });\n"
        + function +
        "var out = [];\n"
        "['random', 'rewired', 'correlation'].forEach(function (graph) {\n"
        "  toasts.length = 0;\n"
        "  var s = settingsFromRun({preprocessing: {graph: graph}, seeds: 5, split: 'varies_with_seed',\n"
        "    training: {epochs: 200, batch_size: 32, learning_rate: 0.001, final_learning_rate: 0.0001,\n"
        "               early_stopping_patience: 20}});\n"
        "  out.push([s.graph, toasts.slice()]);\n"
        "});\n"
        "console.log(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    shown = json.loads(subprocess.run([node, str(script)], capture_output=True, text=True,
                                      encoding="utf-8", check=True).stdout)
    assert shown[0] == ["physics", ["Sensor graph: this run used “random”, which the form does not "
                                    "offer, so the form's default is set instead."]]
    assert shown[1][0] == "physics" and "“rewired”" in shown[1][1][0]
    assert shown[2] == ["correlation", []]


def test_a_graph_that_no_rewiring_can_join_into_one_piece_is_refused():
    """Two separate pairs of sensors: every rewiring of them keeps four sensors with one connection each."""
    pairs = np.zeros((4, 4), dtype=np.float32)
    pairs[0, 1] = pairs[1, 0] = pairs[2, 3] = pairs[3, 2] = 1.0
    with pytest.raises(ValueError, match="in one piece"):
        build_rewired_adjacency(0, adjacency=pairs)
    rewired, _ = build_rewired_adjacency(0, adjacency=pairs, connected=False)
    np.testing.assert_array_equal(rewired.sum(axis=1), [1, 1, 1, 1])
