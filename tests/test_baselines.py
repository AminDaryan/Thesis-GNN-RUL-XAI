"""Checks that the answer file cannot reach a test input, and that scripts/baselines.py refuses what it cannot reproduce.

src/thesis_gnn_rul_xai/evaluation/README.md, under "Ordinary models beside the
network", explains the script and why each of these checks exists.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.config import PreprocessingConfig
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing
from thesis_gnn_rul_xai.evaluation import evaluate, rmse
from thesis_gnn_rul_xai.utils.paths import reports_dir

ROOT = Path(__file__).resolve().parents[1]

# The current default run that the comparison was first measured against;
# any other stored FD001 run with saved networks will do when it is absent.
DEFAULT_RUN = "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f386291-dirty_20260914T150004Z"

# How far every answer in the changed copy of RUL_FD001.txt is moved.
SHIFT = 7


def _script():
    """scripts/baselines.py, imported as a module once and reused."""
    if "baselines_script" in sys.modules:
        return sys.modules["baselines_script"]
    spec = importlib.util.spec_from_file_location("baselines_script", ROOT / "scripts" / "baselines.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, because a dataclass looks its own module up.
    sys.modules["baselines_script"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def changed_answers(raw_dir, tmp_path_factory) -> Path:
    """A copy of the FD001 files in which every answer of RUL_FD001.txt is 7 cycles larger."""
    folder = tmp_path_factory.mktemp("changed_answers")
    for name in ("train_FD001.txt", "test_FD001.txt"):
        shutil.copy(raw_dir / name, folder / name)
    answers = np.loadtxt(raw_dir / "RUL_FD001.txt", dtype=int)
    np.savetxt(folder / "RUL_FD001.txt", answers + SHIFT, fmt="%d")
    return folder


def _stored_run():
    """The default stored run if present, else the newest FD001 run with saved networks."""
    pytest.importorskip("torch")
    from thesis_gnn_rul_xai.training import find_runs, read_run

    if (reports_dir() / DEFAULT_RUN / "seed_42" / "model.pt").is_file():
        return read_run(DEFAULT_RUN)
    runs = [run for run in find_runs(with_weights=True) if run.preprocessing.subset == "FD001"]
    if not runs:
        pytest.skip("no stored FD001 run with saved networks in reports/")
    return runs[-1]


# --- the answer file cannot leak --------------------------------------


@pytest.mark.parametrize("graph", ["physics", "correlation"])
def test_changing_the_answer_file_changes_no_input_and_only_the_test_labels(raw_dir, changed_answers, graph):
    """Every input, every training label, the scaling and the graph stay identical; only the test labels move."""
    from dataclasses import replace

    # The correlation graph is measured from the data, so it is checked too.
    settings = replace(PreprocessingConfig(), graph=graph)
    original = run_preprocessing(settings, data_dir=raw_dir)
    moved = run_preprocessing(settings, data_dir=changed_answers)

    for name in ("train", "validation", "test"):
        before, after = getattr(original, name), getattr(moved, name)
        np.testing.assert_array_equal(before.features, after.features)
        np.testing.assert_array_equal(before.units, after.units)
        np.testing.assert_array_equal(before.end_cycles, after.end_cycles)
    np.testing.assert_array_equal(original.train.targets, moved.train.targets)
    np.testing.assert_array_equal(original.validation.targets, moved.validation.targets)
    pd.testing.assert_frame_equal(original.normalizer.statistics_, moved.normalizer.statistics_)
    np.testing.assert_array_equal(original.adjacency_normalized, moved.adjacency_normalized)

    # The test labels are the answers capped at 125, so an answer already at
    # or above the ceiling cannot move, and one just below it moves part-way.
    answers = np.loadtxt(raw_dir / "RUL_FD001.txt")
    moved_by = moved.test.targets - original.test.targets
    np.testing.assert_array_equal(moved_by, np.minimum(answers + SHIFT, 125) - np.minimum(answers, 125))
    assert [(moved_by == SHIFT).sum(), ((moved_by > 0) & (moved_by < SHIFT)).sum(), (moved_by == 0).sum()] == [86, 3, 11]


def test_a_stored_network_predicts_the_same_whatever_the_answer_file_says(raw_dir, changed_answers):
    """Rebuilt from either answer file, a stored network gives identical predictions; only its score moves."""
    torch = pytest.importorskip("torch")
    from thesis_gnn_rul_xai.training import load_seed
    from thesis_gnn_rul_xai.training.run import predict

    run = _stored_run()
    seed = run.seeds[0]
    with torch.no_grad():
        model, original = load_seed(run, seed, data_dir=raw_dir)
        before = predict(model, original.test, device="cpu")
        model, moved = load_seed(run, seed, data_dir=changed_answers)
        after = predict(model, moved.test, device="cpu")

    np.testing.assert_array_equal(original.test.features, moved.test.features)
    np.testing.assert_array_equal(before, after)
    assert rmse(original.test.targets, before) != rmse(moved.test.targets, after)


# --- scripts/baselines.py ---------------------------------------------


def test_the_short_name_drops_the_commit_and_the_time():
    script = _script()
    assert script.short_name(DEFAULT_RUN) == "fd001_physics_row_cap_5seed_varsplit_cfgba82e3"
    assert script.short_name("not_a_run_name") == "not_a_run_name"


def test_the_fourteen_numbers_are_the_last_reading_of_each_window(raw_dir):
    """Index -1 of the window, the reading the default network receives; index 0 is 29 cycles older."""
    script = _script()
    prepared = run_preprocessing(PreprocessingConfig(), data_dir=raw_dir)
    last = script.model_inputs(prepared.test, "last")
    np.testing.assert_array_equal(last, prepared.test.features[:, :, -1])
    assert not np.array_equal(last, prepared.test.features[:, :, 0])
    summary = script.model_inputs(prepared.test, "last+mean+slope")
    assert summary.shape == (100, 42)
    np.testing.assert_array_equal(summary[:, :14], last)
    np.testing.assert_allclose(summary[:, 14:28], prepared.test.features.astype(float).mean(axis=2))
    ages = script.model_inputs(prepared.test, "age")
    np.testing.assert_array_equal(ages[:, 0], prepared.test.end_cycles)


def test_a_constant_error_gives_the_same_rmse_on_every_resampling(raw_dir):
    """Errors of exactly 1 on one seed and 2 on another average to 1.5 whichever engines are drawn."""
    script = _script()
    truth = np.minimum(np.loadtxt(raw_dir / "RUL_FD001.txt"), 125)
    both = np.vstack([truth, truth])
    predicted = np.vstack([truth + 1, truth - 2])
    draws = np.random.default_rng(0).integers(0, truth.size, size=(50, truth.size))
    np.testing.assert_allclose(script.resampled_rmse(both, predicted, draws), 1.5)
    # Drawing every engine once gives back each seed's own RMSE, averaged.
    everyone = np.arange(truth.size)[None, :]
    flat = np.vstack([np.full_like(truth, 60.0), np.full_like(truth, 80.0)])
    expected = (rmse(truth, flat[0]) + rmse(truth, flat[1])) / 2
    np.testing.assert_allclose(script.resampled_rmse(both, flat, everyone), [expected])


def test_the_script_reproduces_the_stored_scores_and_writes_its_records(raw_dir, tmp_path):
    """One seed, the two quickest models: the stored network's row is results.csv's, the constant is the mean label."""
    from dataclasses import replace

    script = _script()
    run = _stored_run()
    seed = run.seeds[0]
    status = script.main(["--run", str(run.path), "--seeds", str(seed),
                          "--models", "constant", "linear_last",
                          "--data-dir", str(raw_dir), "--output-root", str(tmp_path)])
    assert status == 0
    (folder,) = list(tmp_path.iterdir())
    assert folder.name.startswith(f"baselines_{script.short_name(run.name)}_seeds{seed}_2of14models_")

    results = pd.read_csv(folder / "results.csv").set_index("model")
    stored = float(run.results.loc[run.results["seed"] == seed, "rmse"].iloc[0])
    assert results.loc["network", "rmse_mean"] == pytest.approx(stored, rel=1e-6)

    prepared = run_preprocessing(replace(run.preprocessing, seed=seed), data_dir=raw_dir)
    mean_label = float(prepared.train.targets.astype(float).mean())
    constant = evaluate(prepared.test.targets, np.full(len(prepared.test), mean_label))["rmse"]
    assert results.loc["constant", "rmse_mean"] == pytest.approx(constant, rel=1e-9)
    assert results.loc["constant", "network_minus_model"] == pytest.approx(stored - constant, rel=1e-6)

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert folder.name.endswith(f"_{summary['commit']}_{summary['finished_utc']}")
    assert summary["source_run"]["name"] == run.name
    assert summary["seeds"] == [seed]
    assert summary["environment"]["packages"]["scikit-learn"]
    assert summary["models"]["linear_last"]["inputs"] == "last"
    table = (folder / "table.md").read_text(encoding="utf-8")
    assert "own RMSE [" in table
    assert sum(f"| {better}/1 |" in table for better in (0, 1)) >= 1
    predictions = pd.read_csv(folder / "predictions.csv")
    assert set(predictions["model"]) == {"network", "constant", "linear_last"}
    assert predictions["predicted"].between(0, run.preprocessing.rul_max).all()

    # The clip never touches the stored network, and can only lower an ordinary model's RMSE.
    per_seed = pd.read_csv(folder / "per_seed.csv").set_index("model")
    assert per_seed.loc["network", "n_clipped"] == 0
    assert per_seed.loc["network", "rmse_unclipped"] == pytest.approx(per_seed.loc["network", "rmse"], rel=1e-12)
    for model in ("constant", "linear_last"):
        assert per_seed.loc[model, "rmse_unclipped"] >= per_seed.loc[model, "rmse"] - 1e-12
        if per_seed.loc[model, "n_clipped"] == 0:
            assert per_seed.loc[model, "rmse_unclipped"] == pytest.approx(per_seed.loc[model, "rmse"], rel=1e-12)


def _copy_one_seed(run, seed: int, into: Path) -> Path:
    """A copy of one seed of a stored run, with the run's own records beside it."""
    copy = into / run.name
    (copy / f"seed_{seed}").mkdir(parents=True)
    for name in ("summary.json", "results.csv", "preprocessing_config.yaml", "model_config.yaml"):
        if (run.path / name).is_file():
            shutil.copy(run.path / name, copy / name)
    for name in ("model.pt", "test_predictions.csv", "split.json", "scaling.csv"):
        if (run.path / f"seed_{seed}" / name).is_file():
            shutil.copy(run.path / f"seed_{seed}" / name, copy / f"seed_{seed}" / name)
    return copy


def test_the_script_refuses_a_run_whose_records_do_not_reproduce(raw_dir, tmp_path, capsys):
    """A results.csv that disagrees with the saved predictions, or predictions the network does not give, stop it."""
    script = _script()
    run = _stored_run()
    seed = run.seeds[0]
    copy = _copy_one_seed(run, seed, tmp_path / "runs")
    output = tmp_path / "out"
    arguments = ["--run", str(copy), "--seeds", str(seed), "--models", "constant",
                 "--data-dir", str(raw_dir), "--output-root", str(output)]

    results = pd.read_csv(copy / "results.csv")
    edited = results.copy()
    edited.loc[edited["seed"] == seed, "rmse"] += 0.01
    edited.to_csv(copy / "results.csv", index=False)
    assert script.main(arguments) == 2
    assert "rmse of the saved predictions" in capsys.readouterr().err
    assert not output.exists()

    # Now the predictions are moved and results.csv is rewritten to match them,
    # so only reloading the network can tell.
    saved_path = copy / f"seed_{seed}" / "test_predictions.csv"
    saved = pd.read_csv(saved_path)
    saved.loc[0, "predicted"] += 0.5
    saved.to_csv(saved_path, index=False)
    rescored = evaluate(saved["rul"].to_numpy(float), saved["predicted"].to_numpy(float))
    edited = results.copy()
    for measure in ("rmse", "score", "r2", "asue"):
        edited.loc[edited["seed"] == seed, measure] = rescored[measure]
    edited.to_csv(copy / "results.csv", index=False)
    assert script.main(arguments) == 2
    assert "reloaded network's predictions differ" in capsys.readouterr().err
    assert not output.exists()


def test_the_script_refuses_a_network_filed_under_the_wrong_seed(raw_dir, tmp_path, capsys):
    """One seed's network and records filed as another seed's: every score agrees, the reloaded network does not."""
    script = _script()
    run = _stored_run()
    if len(run.seeds) < 2:
        pytest.skip(f"{run.name} trained one seed only")
    first, second = run.seeds[:2]
    copy = _copy_one_seed(run, first, tmp_path / "runs")
    (copy / f"seed_{first}").rename(copy / f"seed_{second}")
    for name in ("split.json", "scaling.csv"):
        (copy / f"seed_{second}" / name).unlink(missing_ok=True)
    # results.csv now says the second seed scored what the first one did, so the
    # saved predictions rescore correctly and only the split can give it away.
    results = pd.read_csv(copy / "results.csv")
    for measure in ("rmse", "score", "r2", "asue"):
        results.loc[results["seed"] == second, measure] = float(
            results.loc[results["seed"] == first, measure].iloc[0]
        )
    results.to_csv(copy / "results.csv", index=False)
    output = tmp_path / "out"
    status = script.main(["--run", str(copy), "--seeds", str(second), "--models", "constant",
                          "--data-dir", str(raw_dir), "--output-root", str(output)])
    assert status == 2
    error = capsys.readouterr().err
    assert "reloaded network's predictions differ" in error
    assert "rmse of the saved predictions" not in error
    assert not output.exists()


def test_the_script_checks_a_saved_split_and_scaling_when_the_run_has_them(raw_dir, tmp_path, capsys):
    """Untouched records pass; one engine moved in split.json, or one scaling number changed, is refused."""
    pytest.importorskip("torch")
    from thesis_gnn_rul_xai.training import find_runs

    script = _script()
    runs = [run for run in find_runs(with_weights=True)
            if run.preprocessing.subset == "FD001"
            and (run.path / f"seed_{run.seeds[0]}" / "split.json").is_file()
            and (run.path / f"seed_{run.seeds[0]}" / "scaling.csv").is_file()]
    if not runs:
        pytest.skip("no stored FD001 run in reports/ saved its split and its scaling")
    # The default configuration when one of its runs saved both, as the
    # comparison is made on it; otherwise the newest run that did.
    default = [run for run in runs
               if script.short_name(run.name) == script.short_name(DEFAULT_RUN)]
    run = (default or runs)[-1]
    seed = run.seeds[0]

    def attempt(name: str) -> int:
        return script.main(["--run", str(copy), "--seeds", str(seed), "--models", "constant",
                            "--data-dir", str(raw_dir), "--output-root", str(tmp_path / name)])

    copy = _copy_one_seed(run, seed, tmp_path / "runs")
    assert attempt("untouched") == 0
    capsys.readouterr()

    split_path = copy / f"seed_{seed}" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    moved = split["train"].pop(0)
    split["validation"] = sorted([*split["validation"], moved])
    split_path.write_text(json.dumps(split), encoding="utf-8")
    assert attempt("moved_engine") == 2
    assert "the train engines differ from split.json" in capsys.readouterr().err
    assert not (tmp_path / "moved_engine").exists()

    shutil.copy(run.path / f"seed_{seed}" / "split.json", split_path)
    scaling_path = copy / f"seed_{seed}" / "scaling.csv"
    scaling = pd.read_csv(scaling_path)
    scaling.loc[0, "mean"] += 0.001
    scaling.to_csv(scaling_path, index=False)
    assert attempt("changed_scaling") == 2
    assert "the rebuilt scaling differs from scaling.csv" in capsys.readouterr().err
    assert not (tmp_path / "changed_scaling").exists()


# --- the other refusals, and a second run given with --also ---------------


# The same settings as DEFAULT_RUN with the recurrent stage, trained the same day.
RECURRENT_RUN = "fd001_physics_row_cap_5seed_varsplit_lstm32_cfg72f506_f386291-dirty_20260914T134955Z"


def _refused(script, arguments: list[str], output: Path, capsys) -> str:
    """Run the script, require it to refuse and to write nothing, and return what it said."""
    capsys.readouterr()
    assert script.main([*arguments, "--output-root", str(output)]) == 2
    assert not output.exists()
    return capsys.readouterr().err


def test_the_script_refuses_a_run_that_saved_no_network(raw_dir, tmp_path, capsys):
    """A copy with its saved network removed is refused, since nothing could then confirm its split."""
    script = _script()
    run = _stored_run()
    seed = run.seeds[0]
    copy = _copy_one_seed(run, seed, tmp_path / "runs")
    (copy / f"seed_{seed}" / "model.pt").unlink()
    error = _refused(script, ["--run", str(copy), "--seeds", str(seed), "--models", "constant",
                              "--data-dir", str(raw_dir)], tmp_path / "out", capsys)
    assert "no saved network" in error


def test_the_script_refuses_nasa_files_other_than_the_ones_the_run_read(raw_dir, changed_answers, tmp_path, capsys):
    """A copy recording the real files' fingerprints passes on them and is refused on the changed answers."""
    from thesis_gnn_rul_xai.utils.provenance import file_fingerprints

    script = _script()
    run = _stored_run()
    seed = run.seeds[0]
    copy = _copy_one_seed(run, seed, tmp_path / "runs")
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    summary["data_files"] = file_fingerprints(
        [raw_dir / name for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")]
    )
    (copy / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    arguments = ["--run", str(copy), "--seeds", str(seed), "--models", "constant"]

    error = _refused(script, [*arguments, "--data-dir", str(changed_answers)], tmp_path / "changed", capsys)
    assert "the NASA files differ from the ones this run read" in error
    assert script.main([*arguments, "--data-dir", str(raw_dir), "--output-root", str(tmp_path / "same")]) == 0


def _partner(run, seed: int, into: Path) -> Path:
    """The recurrent-stage run beside the default one when both are present, else a copy of the run itself."""
    if run.name == DEFAULT_RUN and (reports_dir() / RECURRENT_RUN / f"seed_{seed}" / "model.pt").is_file():
        return reports_dir() / RECURRENT_RUN
    return _copy_one_seed(run, seed, into)


def test_a_second_run_given_with_also_gets_its_own_row_and_its_own_table(raw_dir, tmp_path):
    """A run prepared the same way passes the same checks, and each network is set against the other."""
    script = _script()
    run = _stored_run()
    seed = run.seeds[0]
    partner = _partner(run, seed, tmp_path / "partner")
    status = script.main(["--run", str(run.path), "--also", str(partner), "--seeds", str(seed),
                          "--models", "constant", "--data-dir", str(raw_dir),
                          "--output-root", str(tmp_path / "out")])
    assert status == 0
    (folder,) = list((tmp_path / "out").iterdir())

    results = pd.read_csv(folder / "results.csv").set_index(["network", "model"])
    assert sorted(results.index) == [(network, model)
                                     for network in ("network", "network_1")
                                     for model in ("constant", "network", "network_1")]
    # Each network minus the other: the same difference, with its sign turned round.
    assert results.loc[("network", "network_1"), "network_minus_model"] == pytest.approx(
        -results.loc[("network_1", "network"), "network_minus_model"], abs=1e-12)
    stored = float(pd.read_csv(Path(partner) / "results.csv").set_index("seed").loc[seed, "rmse"])
    assert results.loc[("network", "network_1"), "rmse_mean"] == pytest.approx(stored, rel=1e-6)

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert [also["name"] for also in summary["also_runs"]] == [Path(partner).name]
    assert sorted(summary["verification"]["max_reload_gap_cycles"]) == ["network", "network_1"]
    assert (folder / "table.md").read_text(encoding="utf-8").count("## Set against:") == 2


def test_a_second_run_prepared_differently_or_missing_the_seed_is_refused(raw_dir, tmp_path, capsys):
    """An --also copy with one preprocessing setting changed, or without the seed asked for, stops the script."""
    import yaml

    script = _script()
    run = _stored_run()
    seed = run.seeds[0]
    arguments = ["--run", str(run.path), "--seeds", str(seed), "--models", "constant",
                 "--data-dir", str(raw_dir)]

    changed = _copy_one_seed(run, seed, tmp_path / "changed")
    settings_path = changed / "preprocessing_config.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    # A stored run may have smoothing switched off, which is stored as null.
    alpha = settings["smoothing_alpha"]
    settings["smoothing_alpha"] = 0.5 if alpha is None else float(alpha) + 0.1
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    error = _refused(script, [*arguments, "--also", str(changed)], tmp_path / "out_changed", capsys)
    assert "was prepared differently" in error

    missing = _copy_one_seed(run, seed, tmp_path / "missing")
    results = pd.read_csv(missing / "results.csv")
    results.loc[results["seed"] != seed].to_csv(missing / "results.csv", index=False)
    error = _refused(script, [*arguments, "--also", str(missing)], tmp_path / "out_missing", capsys)
    assert f"has no seed [{seed}]" in error


def test_rebuilt_data_that_differ_between_runs_or_between_seeds_are_refused(raw_dir, tmp_path, capsys, monkeypatch):
    """Two checks no edited record reaches, forced by altering the rebuilt data after its own checks have passed."""
    script = _script()
    run = _stored_run()
    if len(run.seeds) < 2:
        pytest.skip(f"{run.name} trained one seed only")
    first, second = run.seeds[:2]
    checked = script.rebuild_and_verify

    def other_test_inputs(stored, seed, data_dir):
        """The --also run's test inputs moved by one unit in one place."""
        prepared, saved, gap = checked(stored, seed, data_dir)
        if Path(stored.path) != Path(run.path):
            features = prepared.test.features.copy()
            features[0, 0, -1] += 1.0
            prepared.test.features = features
        return prepared, saved, gap

    partner = _copy_one_seed(run, first, tmp_path / "partner")
    monkeypatch.setattr(script, "rebuild_and_verify", other_test_inputs)
    error = _refused(script, ["--run", str(run.path), "--also", str(partner), "--seeds", str(first),
                              "--models", "constant", "--data-dir", str(raw_dir)],
                     tmp_path / "out_inputs", capsys)
    assert "rebuilt data differ from" in error

    def reordered_engines(stored, seed, data_dir):
        """The second seed's test engines in reverse order."""
        prepared, saved, gap = checked(stored, seed, data_dir)
        if seed == second:
            prepared.test.units = prepared.test.units[::-1].copy()
        return prepared, saved, gap

    monkeypatch.setattr(script, "rebuild_and_verify", reordered_engines)
    error = _refused(script, ["--run", str(run.path), "--seeds", str(first), str(second),
                              "--models", "constant", "--data-dir", str(raw_dir)],
                     tmp_path / "out_engines", capsys)
    assert "the test engines changed between seeds" in error


# --- the split field, which runs before 2026-10-03 wrote with no split (audit item D-4) ---


def test_two_runs_that_held_nothing_back_are_not_refused_over_their_recorded_split():
    """With nothing held back there is no split to differ in; with engines held back the field must agree."""
    from types import SimpleNamespace

    script = _script()

    def stored(name: str, fraction: float, split: str) -> SimpleNamespace:
        """The parts of a stored run check_same_data reads."""
        return SimpleNamespace(name=name, seeds=[42], summary={"split": split},
                               preprocessing=PreprocessingConfig(validation_fraction=fraction))

    # What a run recorded before 2026-10-03 and what one records since, for the same settings.
    script.check_same_data(stored("before", 0.0, "varies_with_seed"), stored("since", 0.0, "none"), [42])
    with pytest.raises(script.NotReproduced, match="prepared differently"):
        script.check_same_data(stored("varied", 0.2, "varies_with_seed"), stored("held", 0.2, "fixed"), [42])


# --- results.csv carries the early count beside the ASUE (assumption A-39) ---


def test_every_row_of_the_comparison_carries_the_early_count_beside_its_asue():
    """A stored network's seeds 42 and 43, and two flat guesses at the label ceiling and one cycle under it."""
    import warnings

    script = _script()
    folder = reports_dir() / DEFAULT_RUN
    if not (folder / "seed_43" / "test_predictions.csv").is_file():
        pytest.skip(f"the stored run {DEFAULT_RUN} is not in reports/")
    seeds = [42, 43]
    saved = {s: pd.read_csv(folder / f"seed_{s}" / "test_predictions.csv") for s in seeds}
    truth = np.vstack([saved[s]["rul"].to_numpy(float) for s in seeds])
    predicted = {
        "network": np.vstack([saved[s]["predicted"].to_numpy(float) for s in seeds]),
        # 125 is above no capped label, so no engine is early; 124 is early by
        # one cycle on the 11 engines whose labels were capped down to 125.
        "ceiling": np.full_like(truth, 125.0),
        "one_under": np.full_like(truth, 124.0),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # ASUE's warning for the ceiling
        scored = {key: [evaluate(truth[i], guess[i]) for i in range(len(seeds))]
                  for key, guess in predicted.items()}
    rows = {key: {"label": key, "inputs": "test", "seeds": seeds,
                  **{f"{m}_by_seed": [one[m] for one in scored[key]]
                     for m in (*script.MEASURES, "n_early")}}
            for key in predicted}
    draws = np.random.default_rng(0).integers(0, truth.shape[1], size=(20, truth.shape[1]))
    resampled = {key: script.resampled_rmse(truth, guess, draws) for key, guess in predicted.items()}
    results = pd.DataFrame(script.compare("network", rows, resampled)).set_index("model")

    # The network: 47 and 49 of the 100 engines early, ASUE 12.2526 and 10.3136,
    # as the stored run's results.csv has them (53 and 51 late).
    assert [one["n_early"] for one in scored["network"]] == [47, 49]
    assert results.loc["network", "n_early_mean"] == 48.0
    assert results.loc["network", "asue_mean"] == pytest.approx((12.2526 + 10.3136) / 2, abs=1e-4)
    # The ceiling: no early engine on either seed, so its ASUE is not a number,
    # and so is the mean, with the count of 0 beside it saying why.
    assert results.loc["ceiling", "n_early_mean"] == 0.0
    assert np.isnan(results.loc["ceiling", "asue_mean"])
    assert (results.loc["one_under", "n_early_mean"], results.loc["one_under", "asue_mean"]) == (11.0, 1.0)
