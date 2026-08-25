#!/usr/bin/env python3
"""
Data preparation helper
=======================

A small command-line tool with two jobs:

``--check``
    Report which raw C-MAPSS files are present, which are missing, and whether
    the ones present have the number of rows and engines published for the
    official release.

``--run``
    Execute the full preprocessing pipeline and save the result.

Examples
--------
Check what is available::

    python scripts/prepare_data.py --check

Run the pipeline with a specific settings file::

    python scripts/prepare_data.py --run --config config/fd001.yaml

Run it against a copy of the dataset kept outside the project::

    python scripts/prepare_data.py --run --data-dir /path/to/CMAPSSData
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable when this script is run directly from a clone
# that has not been installed with `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_gnn_rul_xai.config import PreprocessingConfig, load_config  # noqa: E402
from thesis_gnn_rul_xai.data.loading import (  # noqa: E402
    DOWNLOAD_URL,
    EXPECTED_SHAPES,
    read_cmapss_frame,
)
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing  # noqa: E402
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging  # noqa: E402
from thesis_gnn_rul_xai.utils.paths import processed_data_dir, raw_data_dir  # noqa: E402


def check_raw_data(subset: str, directory: Path | None = None) -> int:
    """Report the state of the raw data directory. Returns an exit code."""
    directory = directory or raw_data_dir()
    print(f"Looking for {subset} files in: {directory}\n")

    names = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    missing = [name for name in names if not (directory / name).is_file()]

    for name in names:
        path = directory / name
        if path.is_file():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  found    {name:20s} {size_mb:6.2f} MB")
        else:
            print(f"  MISSING  {name}")

    if missing:
        print(
            "\nSome files are missing. Download the "
            "'Turbofan Engine Degradation Simulation Data Set' from the NASA "
            f"Prognostics Center of Excellence repository:\n\n    {DOWNLOAD_URL}\n\n"
            f"and put the three {subset} text files into {directory}."
        )
        return 1

    print("\nAll files present. Checking their size against the published release:")
    train = read_cmapss_frame(directory / f"train_{subset}.txt")
    test = read_cmapss_frame(directory / f"test_{subset}.txt")
    expected = EXPECTED_SHAPES.get(subset, {})
    actual = {
        "train_rows": len(train),
        "train_units": int(train["unit"].nunique()),
        "test_units": int(test["unit"].nunique()),
    }

    all_match = True
    for key, expected_value in expected.items():
        status = "OK " if actual[key] == expected_value else "DIFFERS"
        if actual[key] != expected_value:
            all_match = False
        print(f"  {status:8s} {key:14s} expected {expected_value:>6}, found {actual[key]:>6}")

    if all_match:
        print("\nThe files match the official release.")
        return 0

    print(
        "\nThe files do not match the published counts. They may be a different "
        "version or may have been modified. Investigate before using them for "
        "any reported result."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check, generate, or preprocess C-MAPSS data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--subset", default="FD001", help="Sub-dataset name (default: FD001)")
    parser.add_argument("--check", action="store_true", help="Report on the raw data files")
    parser.add_argument("--run", action="store_true", help="Run the preprocessing pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML settings file")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory holding the C-MAPSS text files (default: data/raw)",
    )
    parser.add_argument("--quiet", action="store_true", help="Show warnings only")
    arguments = parser.parse_args()

    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

    if not any((arguments.check, arguments.run)):
        parser.print_help()
        return 0

    exit_code = 0

    if arguments.check:
        exit_code |= check_raw_data(arguments.subset, arguments.data_dir)

    if arguments.run:
        config = load_config(arguments.config) if arguments.config else PreprocessingConfig()
        prepared = run_preprocessing(config, data_dir=arguments.data_dir)
        print()
        print(prepared.summary().to_string())
        destination = prepared.save(processed_data_dir())
        print(f"\nSaved to {destination}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
