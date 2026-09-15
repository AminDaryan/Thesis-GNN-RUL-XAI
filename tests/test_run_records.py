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

