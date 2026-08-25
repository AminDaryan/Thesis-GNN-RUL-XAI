#!/usr/bin/env python3
"""
Data preparation helper
=======================

A small command-line tool with three jobs:

``--check``
    Report which raw C-MAPSS files are present, which are missing, and whether
    the ones present have the number of rows and engines published for the
    official release.

``--synthetic``
    Generate artificial data in the same format as the real files, so the
    pipeline can be run and tested without the NASA download.

``--run``
    Execute the full preprocessing pipeline and save the result.

Examples
--------
Check what is available::

    python scripts/prepare_data.py --check

Create example data and run the pipeline on it::

    python scripts/prepare_data.py --synthetic --run --use-synthetic

Run the pipeline on the real files with a specific settings file::

    python scripts/prepare_data.py --run --config config/fd001.yaml
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
from thesis_gnn_rul_xai.data.loading import EXPECTED_SHAPES, read_cmapss_frame  # noqa: E402
from thesis_gnn_rul_xai.data.pipeline import run_preprocessing  # noqa: E402
from thesis_gnn_rul_xai.data.synthetic import write_synthetic_subset  # noqa: E402
from thesis_gnn_rul_xai.utils.logging_setup import configure_logging  # noqa: E402
from thesis_gnn_rul_xai.utils.paths import (  # noqa: E402
    processed_data_dir,
    raw_data_dir,
    synthetic_data_dir,
)

DOWNLOAD_URL = (
    "https://www.nasa.gov/intelligent-systems-division/"
    "discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
)


def check_raw_data(subset: str) -> int:
    """Report the state of the raw data directory. Returns an exit code."""
    directory = raw_data_dir()
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
            "'Turbofan Engine Degradation Simulation' dataset from the NASA "
            f"Prognostics Center of Excellence repository:\n  {DOWNLOAD_URL}\n"
            f"and place the three {subset} text files in {directory}.\n\n"
            "To work without the real data for now, run this script with "
            "--synthetic and set use_synthetic_data: true in your config file."
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
    parser.add_argument("--synthetic", action="store_true", help="Generate example data")
    parser.add_argument("--run", action="store_true", help="Run the preprocessing pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML settings file")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Force the pipeline to read the generated example data",
    )
    parser.add_argument("--quiet", action="store_true", help="Show warnings only")
    arguments = parser.parse_args()

    configure_logging(logging.WARNING if arguments.quiet else logging.INFO)

    if not any((arguments.check, arguments.synthetic, arguments.run)):
        parser.print_help()
        return 0

    exit_code = 0

    if arguments.check:
        exit_code |= check_raw_data(arguments.subset)

    if arguments.synthetic:
        paths = write_synthetic_subset(synthetic_data_dir(), arguments.subset)
        print(f"\nGenerated example data in {synthetic_data_dir()}:")
        for name, path in paths.items():
            print(f"  {name:6s} {path.name}")
        print(
            "\nReminder: this data is artificial. Use it to test the code, "
            "never to produce a reported result."
        )

    if arguments.run:
        config = load_config(arguments.config) if arguments.config else PreprocessingConfig()
        if arguments.use_synthetic:
            config.use_synthetic_data = True
        prepared = run_preprocessing(config)
        print()
        print(prepared.summary().to_string())
        # Generated data never shares a directory with real results. Without
        # this, a synthetic run would overwrite a real one under identical
        # file names, and nothing in the file names would say which is which.
        output_dir = processed_data_dir()
        if config.use_synthetic_data:
            output_dir = output_dir / "synthetic_demo"
            print(
                "Running on generated data, so the output goes to a separate "
                "folder to keep it away from anything that could be mistaken "
                "for a result."
            )

        destination = prepared.save(output_dir)
        print(f"\nSaved to {destination}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
