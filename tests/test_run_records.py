"""Checks that a stored run records how it was made, and that today's code still reproduces it.

src/thesis_gnn_rul_xai/training/README.md explains what a stored run holds and
why a run that no longer reproduces must be trained again rather than quoted.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_gnn_rul_xai.utils.paths import reports_dir
from thesis_gnn_rul_xai.utils.provenance import (
    capture_code_state,
    environment,
    file_fingerprints,
    write_code_state,
)

# A prediction is saved to a CSV file with a handful of decimals, so a reload
# can differ from it by rounding; anything this large is a different network.
PREDICTION_TOLERANCE = 1e-3


def test_the_environment_names_python_and_every_library_a_number_depends_on():
    recorded = environment()
    assert recorded["python"].count(".") == 2
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "torch"):
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        assert recorded["packages"][package] == installed
    assert recorded["platform"]


def test_a_fingerprint_is_the_sha256_of_the_file_bytes(tmp_path):
    """The known answer for the three bytes "abc", from the SHA-256 standard's first example."""
    file = tmp_path / "abc.txt"
    file.write_bytes(b"abc")
    assert file_fingerprints([file]) == {
        "abc.txt": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    }


def test_the_code_state_is_written_only_when_the_code_differs_from_the_commit(tmp_path):
    state = capture_code_state()
    written = write_code_state(tmp_path, state)
    assert written == list(state["status"])
    assert (tmp_path / "code_state").is_dir() == bool(state["status"])
    if state["status"]:
        assert (tmp_path / "code_state" / "uncommitted.patch").read_text(encoding="utf-8") == state["patch"]


def _stored_runs() -> list[Path]:
    """Every run in reports/ that saved each seed's network and test predictions."""
    root = reports_dir()
    runs = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        folders = summary.get("seed_folders") or []
        if folders and all((path / folder / "model.pt").is_file()
                           and (path / folder / "test_predictions.csv").is_file()
                           for folder in folders):
            runs.append(path)
    return runs


STORED = _stored_runs()


@pytest.mark.skipif(not STORED, reason="no stored run with saved networks in reports/")
@pytest.mark.parametrize("path", STORED, ids=[path.name for path in STORED])
def test_a_stored_network_still_gives_the_predictions_it_saved(path, raw_dir):
    """Rebuild each seed's data with today's code, reload its network, and compare."""
    torch = pytest.importorskip("torch")
    from thesis_gnn_rul_xai.training import load_seed, read_run
    from thesis_gnn_rul_xai.training.run import predict

    run = read_run(path)
    recorded_data = run.summary.get("data_files")
    if recorded_data:
        today = file_fingerprints([raw_dir / name for name in recorded_data])
        assert today == recorded_data, "the NASA files differ from the ones this run read"

    for seed in run.seeds:
        folder = path / f"seed_{seed}"
        with torch.no_grad():
            model, prepared = load_seed(run, seed, data_dir=raw_dir)
            predicted = predict(model, prepared.test, device="cpu")
        saved = pd.read_csv(folder / "test_predictions.csv")
        np.testing.assert_array_equal(saved["unit"].to_numpy(), prepared.test.units)
        np.testing.assert_allclose(saved["rul"].to_numpy(), prepared.test.targets)
        gap = float(np.abs(predicted - saved["predicted"].to_numpy()).max())
        assert gap < PREDICTION_TOLERANCE, f"seed {seed}: predictions moved by {gap:.2e} cycles"

        if (folder / "split.json").is_file():
            split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
            for name in ("train", "validation", "test"):
                rebuilt = sorted(int(u) for u in set(getattr(prepared, name).units.tolist()))
                assert rebuilt == split[name], f"seed {seed}: the {name} engines differ"
        if (folder / "scaling.csv").is_file():
            stored = pd.read_csv(folder / "scaling.csv")
            rebuilt = prepared.normalizer.statistics_.reset_index(drop=True)
            pd.testing.assert_frame_equal(rebuilt, stored, check_dtype=False, rtol=1e-12)


def test_a_new_run_records_its_command_environment_data_and_split():
    """Runs written since these records were added must carry every one of them."""
    newest = [path for path in STORED
              if "data_files" in json.loads((path / "summary.json").read_text(encoding="utf-8"))]
    if not newest:
        pytest.skip("no stored run written since runs began recording their data files")
    for path in newest:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        assert summary["command"][0] == "train.py"
        assert set(summary["environment"]["packages"]) >= {"numpy", "torch"}
        assert all(len(value) == 64 for value in summary["data_files"].values())
        # "-dirty" also counts edited documents, which are not code, so only a
        # run that lists changed code files has to carry the patch.
        if summary["uncommitted_code"]:
            assert summary["commit"].endswith("-dirty")
            assert (path / "code_state" / "uncommitted.patch").is_file()
        for folder in summary["seed_folders"]:
            assert (path / folder / "split.json").is_file()
            assert (path / folder / "scaling.csv").is_file()


# --- The "cfg" part of a run's folder name ------------------------------------
# training/README.md explains the digest, and the change of 2026-10-03 that
# took the prose notes out of it.


def test_two_configurations_that_differ_only_in_their_notes_share_a_digest():
    """Notes are prose: rewriting them must not rename a run."""
    from dataclasses import replace

    from thesis_gnn_rul_xai.config import PreprocessingConfig, load_config

    default = PreprocessingConfig()
    assert replace(default, notes="Any other sentence at all.").digest() == default.digest()
    assert replace(default, notes="").digest() == default.digest()
    # The case that was found: the same eleven settings, two different sentences.
    from_file = load_config(Path(__file__).resolve().parents[1] / "config" / "fd001.yaml")
    assert from_file.notes != default.notes
    assert from_file.digest() == default.digest() == "818b40"


def test_a_change_to_any_setting_changes_the_digest():
    """Every one of the eleven settings is inside the digest."""
    from dataclasses import replace

    from thesis_gnn_rul_xai.config import PreprocessingConfig, settings_without_notes

    default = PreprocessingConfig()
    changed = {
        "subset": "FD002", "window_size": 31, "window_stride": 2, "rul_max": 130,
        "clip_test_labels": False, "normalization": "minmax_global", "smoothing_alpha": None,
        "graph": "complete", "adjacency_normalization": "symmetric",
        "validation_fraction": 0.0, "seed": 43,
    }
    assert set(changed) == set(settings_without_notes(default.to_dict()))
    digests = {key: replace(default, **{key: value}).digest() for key, value in changed.items()}
    assert default.digest() not in digests.values()
    assert len(set(digests.values())) == len(changed)


def test_the_training_script_and_the_run_finder_use_the_shared_rule():
    """train.py names a run with PreprocessingConfig.digest(), and hashes no preprocessing settings itself."""
    import ast

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts" / "train.py").read_text(encoding="utf-8"))
    assigned = [
        ast.unparse(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "config_digest" for target in node.targets)
    ]
    assert assigned == ["preprocessing.digest()"]
    hashed = [ast.unparse(node) for node in ast.walk(tree)
              if isinstance(node, ast.Call) and ast.unparse(node.func) == "hashlib.sha256"]
    assert hashed and not any("preprocessing" in call for call in hashed)
    # The run finder, the playground's comparison of two runs, the baselines
    # script's check of a second run and the claim checker's store of trained
    # networks leave the notes out with the same function.
    for module in ("src/thesis_gnn_rul_xai/training/stored.py",
                   "src/thesis_gnn_rul_xai/docsite/playground/runs.py",
                   "scripts/baselines.py",
                   "scripts/verify_claims.py"):
        source = (root / module).read_text(encoding="utf-8")
        assert "settings_without_notes(" in source and '"notes"' not in source, module


def test_a_stored_run_name_carries_the_digest_of_the_settings_it_saved():
    """Code from before 2026-10-03 hashed the notes as well, code since leaves them out; each name fits one rule."""
    import hashlib
    import re

    from thesis_gnn_rul_xai.config import load_config

    root = reports_dir()
    counted = {"notes hashed": 0, "notes left out": 0}
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        named = re.search(r"_cfg([0-9a-f]{6})_", path.name)
        if not named or not (path / "preprocessing_config.yaml").is_file():
            continue
        try:
            config = load_config(path / "preprocessing_config.yaml")
        except (ValueError, TypeError):
            continue  # a setting renamed since; find_runs skips such a run too
        if (path / "summary.json").is_file():  # absent while a run is still training
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            assert summary.get("config_digest", named.group(1)) == named.group(1), path.name
        if named.group(1) == config.digest():
            counted["notes left out"] += 1
            continue
        # The rule scripts/train.py used until 2026-10-03, kept here only to read old names.
        before = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:6]
        assert named.group(1) == before, path.name
        counted["notes hashed"] += 1
    if not any(counted.values()):
        pytest.skip("no stored run with a configuration digest in its name in reports/")


def test_a_number_written_as_a_whole_number_or_a_decimal_gives_one_digest():
    """0 and 0.0 are equal to find_runs, so they must be equal to the digest too."""
    from thesis_gnn_rul_xai.config import PreprocessingConfig, settings_without_notes

    pairs = [
        ({"validation_fraction": 0}, {"validation_fraction": 0.0}),
        ({"smoothing_alpha": 1}, {"smoothing_alpha": 1.0}),
        ({"rul_max": 125.0}, {"rul_max": 125}),
        ({"window_size": 30.0}, {"window_size": 30}),
        ({"clip_test_labels": 1}, {"clip_test_labels": True}),
    ]
    for written, declared in pairs:
        first, second = PreprocessingConfig(**written), PreprocessingConfig(**declared)
        assert settings_without_notes(first.to_dict()) == settings_without_notes(second.to_dict())
        assert first.digest() == second.digest(), written
        (name,) = written
        assert type(getattr(first, name)) is type(declared[name]), name
    # Measured 2026-10-03; before the types were made to agree, validation_fraction
    # 0 gave e950d9 and 0.0 gave 2c9b5b.
    assert PreprocessingConfig(validation_fraction=0).digest() == "2c9b5b"
    assert PreprocessingConfig(smoothing_alpha=1).digest() == "8ea41a"


# --- The split a run records (audit item D-4, changed 2026-10-03) -------------
# training/README.md, "The split part of a run folder's name", explains the three
# values and why a run stored before the change keeps the one it wrote.

HELD_BACK_RUN = "fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z"
NOTHING_HELD_BACK_RUN = "fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_19dceb5_20261002T183904Z"


def test_a_stored_run_of_each_kind_of_split_still_loads_and_is_listed_as_it_happened(tmp_path, raw_dir):
    """A run that held engines back, one that held none back but recorded "varies_with_seed", and a copy of it recording "none"."""
    torch = pytest.importorskip("torch")
    import shutil

    from thesis_gnn_rul_xai.docsite.playground.runs import list_runs
    from thesis_gnn_rul_xai.training import find_runs, load_seed, read_run
    from thesis_gnn_rul_xai.training.run import predict

    sources = {name: reports_dir() / name for name in (HELD_BACK_RUN, NOTHING_HELD_BACK_RUN)}
    if not all((path / "seed_42" / "model.pt").is_file() for path in sources.values()):
        pytest.skip("the two stored runs this test names are not in reports/")
    # A copy of each run's records, never the run itself (working rule 3a); the
    # third copy records "none", as a run with nothing held back does since the
    # change, and carries seed 42's network so it can be loaded.
    copies = tmp_path / "reports"
    renamed = NOTHING_HELD_BACK_RUN.replace("_varsplit_", "_nosplit_")
    for name, source in [*sources.items(), (renamed, sources[NOTHING_HELD_BACK_RUN])]:
        (copies / name).mkdir(parents=True)
        for record in ("summary.json", "results.csv", "preprocessing_config.yaml", "model_config.yaml"):
            shutil.copy(source / record, copies / name / record)
    shutil.copytree(sources[NOTHING_HELD_BACK_RUN] / "seed_42", copies / renamed / "seed_42")
    summary = json.loads((copies / renamed / "summary.json").read_text(encoding="utf-8"))
    (copies / renamed / "summary.json").write_text(json.dumps({**summary, "split": "none"}), encoding="utf-8")

    # Recorded split, and how many engines seed 42 held back once rebuilt.
    expected = {HELD_BACK_RUN: ("varies_with_seed", 20),
                NOTHING_HELD_BACK_RUN: ("varies_with_seed", 0),
                renamed: ("none", 0)}
    for path in (sources[HELD_BACK_RUN], sources[NOTHING_HELD_BACK_RUN], copies / renamed):
        run = read_run(path)
        recorded, held_back = expected[path.name]
        assert run.summary["split"] == recorded
        with torch.no_grad():
            model, prepared = load_seed(run, 42, data_dir=raw_dir)
            predicted = predict(model, prepared.test, device="cpu")
        saved = pd.read_csv(path / "seed_42" / "test_predictions.csv")
        assert float(np.abs(predicted - saved["predicted"].to_numpy()).max()) < PREDICTION_TOLERANCE
        assert len(set(prepared.validation.units.tolist())) == held_back

    # The run finder matches the field as written; the playground lists what happened.
    assert [run.name for run in find_runs(root=copies, split="none")] == [renamed]
    listed = {run["name"]: run["split"] for run in list_runs(tmp_path)}
    assert listed == {HELD_BACK_RUN: "varies_with_seed", NOTHING_HELD_BACK_RUN: "none", renamed: "none"}

