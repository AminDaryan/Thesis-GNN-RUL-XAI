"""Runs every preparation step in order and hands back arrays ready for training.

Read README.md in this folder. It walks through the steps one at a time and
explains why they are in this order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import PreprocessingConfig
from ...graph.adjacency import (
    adjacency_report,
    build_fully_connected_adjacency,
    build_physics_adjacency,
    normalize_adjacency,
)
from ...utils.logging_setup import get_logger
from ...utils.paths import processed_data_dir, raw_data_dir
from ..labeling import apply_piecewise_rul
from ..loading import CMAPSSData, load_subset
from ..normalization import OperatingConditionEncoder, SensorNormalizer
from ..sensors import (
    PAPER_SENSOR_COLUMNS,
    PAPER_SENSOR_SYMBOLS,
    UNIT_COL,
    detect_uninformative_sensors,
)
from ..smoothing import apply_ewma
from ..windowing import (
    WindowedDataset,
    make_last_windows,
    make_windows,
    split_units_by_id,
)

__all__ = ["PreparedData", "run_preprocessing"]

logger = get_logger(__name__)


@dataclass
class PreparedData:
    """Everything the training stage needs, plus everything needed to check it.

    README.md in this folder lists every field.
    """

    config: PreprocessingConfig
    train: WindowedDataset
    validation: WindowedDataset
    test: WindowedDataset
    adjacency_raw: np.ndarray
    adjacency_normalized: np.ndarray
    sensor_symbols: tuple[str, ...]
    normalizer: SensorNormalizer
    condition_encoder: OperatingConditionEncoder
    diagnostics: dict[str, object]
    raw: CMAPSSData | None = None
    frames: dict[str, pd.DataFrame] | None = None

    def summary(self) -> pd.DataFrame:
        """A small table describing the three sets, to look at."""
        rows = []
        for name, dataset in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            rows.append(
                {
                    "split": name,
                    "examples": len(dataset),
                    "engines": len(np.unique(dataset.units)) if len(dataset) else 0,
                    "feature_shape": str(tuple(dataset.features.shape[1:])),
                    "target_min": float(dataset.targets.min()) if len(dataset) else float("nan"),
                    "target_max": float(dataset.targets.max()) if len(dataset) else float("nan"),
                }
            )
        return pd.DataFrame(rows).set_index("split")

    def save(self, directory: str | Path | None = None) -> Path:
        """Write five files: the arrays, the graph, the scaling numbers, the
        settings, and what was observed on the way. README.md names them all.
        """
        directory = Path(directory) if directory is not None else processed_data_dir()
        directory.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            directory / "windows.npz",
            **{
                f"{split}_{field}": getattr(dataset, field)
                for split, dataset in (
                    ("train", self.train),
                    ("val", self.validation),
                    ("test", self.test),
                )
                for field in ("features", "targets", "units", "end_cycles")
            },
            sensor_columns=np.array(self.train.sensor_columns),
        )
        np.savez_compressed(
            directory / "adjacency.npz",
            adjacency_raw=self.adjacency_raw,
            adjacency_normalized=self.adjacency_normalized,
            sensor_symbols=np.array(self.sensor_symbols),
        )
        if self.normalizer.statistics_ is not None:
            self.normalizer.statistics_.to_csv(
                directory / "normalization_statistics.csv", index=False
            )
        self.config.save(directory / "config.yaml")
        (directory / "diagnostics.json").write_text(
            json.dumps(self.diagnostics, indent=2, default=str), encoding="utf-8"
        )

        logger.info("Saved prepared data to %s", directory)
        return directory


def run_preprocessing(
    config: PreprocessingConfig | None = None,
    *,
    data_dir: str | Path | None = None,
    keep_frames: bool = False,
) -> PreparedData:
    """Run every preparation step in order and return the result.

    Set keep_frames=True to keep the intermediate tables as well, which costs
    memory but is what a notebook wants.
    """
    config = config or PreprocessingConfig()

    data_dir = Path(data_dir) if data_dir is not None else raw_data_dir()

    diagnostics: dict[str, object] = {
        "subset": config.subset,
        "data_dir": str(data_dir),
    }

    # -- 1. load ----------------------------------------------------------
    data = load_subset(config.subset, data_dir=data_dir)

    # -- 2. diagnose the sensor selection ---------------------------------
    detected = detect_uninformative_sensors(data.train)
    uninformative = set(detected["constant"]) | set(detected["low_resolution"])
    kept_but_uninformative = sorted(uninformative & set(PAPER_SENSOR_SYMBOLS))
    diagnostics["sensor_diagnostics"] = detected
    diagnostics["kept_but_uninformative"] = kept_but_uninformative
    if kept_but_uninformative:
        logger.warning(
            "The paper keeps sensor(s) %s, but in this data they carry no "
            "usable variation. They are kept anyway, because the adjacency "
            "matrix is defined over the paper's fixed 14-sensor set. Report "
            "this observation rather than silently changing the selection.",
            kept_but_uninformative,
        )

    # -- 3 & 4. pick the sensors and cap the labels ------------------------
    # Fig. 1 of the paper caps the labels after smoothing; here it happens
    # before. That cannot change anything: capping touches only the label
    # column and smoothing touches only the sensor columns, so neither reads
    # what the other writes. Doing it here keeps the label work together.
    columns = list(PAPER_SENSOR_COLUMNS)
    train = apply_piecewise_rul(data.train, rul_max=config.rul_max)
    test = apply_piecewise_rul(
        data.test, rul_max=config.rul_max
    ) if config.clip_test_labels else data.test.copy()

    # -- 5. smooth --------------------------------------------------------
    train = apply_ewma(train, columns, alpha=config.smoothing_alpha)
    test = apply_ewma(test, columns, alpha=config.smoothing_alpha)

    # -- 6. hold some engines back ----------------------------------------
    # ASSUMPTION A-25: split first, then fit the scaling on the training
    # engines only. Fig. 1 of the paper prints the reverse order -- data
    # normalisation, then train and test datasets -- and never says which
    # engines the statistics are fitted on. This happens BEFORE anything is
    # measured from the data: if it happened later, the held-back engines
    # would help decide the mean, the standard deviation and so on, and their
    # error would then look better than it really is -- the same problem that
    # keeping the test engines separate exists to avoid. Written up in
    # docs/ASSUMPTIONS.md.
    train_units, validation_units = split_units_by_id(
        np.unique(train[UNIT_COL].to_numpy()),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    # -- 7. operating conditions, then rescaling ---------------------------
    # Measured on the training engines only, then applied unchanged to the
    # held-back and test engines.
    training_rows = train[UNIT_COL].isin(train_units)
    encoder = OperatingConditionEncoder(seed=config.seed).fit(train[training_rows])
    train = encoder.transform(train)
    test = encoder.transform(test)
    diagnostics["n_operating_conditions"] = encoder.n_detected

    # Worked out again rather than reused: `train` now points at a different
    # table than it did above, so the old mask would line up with nothing.
    training_rows = train[UNIT_COL].isin(train_units)
    normalizer = SensorNormalizer(method=config.normalization, columns=columns).fit(
        train[training_rows]
    )
    train = normalizer.transform(train)
    test = normalizer.transform(test)

    # -- 8. windowing -----------------------------------------------------
    train_windows = make_windows(
        train,
        columns,
        window_size=config.window_size,
        stride=config.window_stride,
    )
    test_windows = make_last_windows(
        test,
        columns,
        window_size=config.window_size,
    )

    # -- 9. apply the split decided in step 6 ------------------------------
    validation_windows = train_windows.subset_by_units(validation_units)
    train_windows = train_windows.subset_by_units(train_units)

    overlap = set(train_windows.units.tolist()) & set(validation_windows.units.tolist())
    if overlap:  # pragma: no cover - defensive
        raise AssertionError(
            f"Engines {sorted(overlap)} appear in both the training and the "
            "validation split. This would invalidate the experiment."
        )

    # -- the sensor graph -------------------------------------------------
    # Which graph, and how it is rescaled, are settings rather than fixed
    # values, so that the graph saved next to a result is the one the network
    # was actually given. Before that was true, a run could store one graph
    # and train on another, and nothing would say so.
    if config.graph == "physics":
        adjacency_raw, symbols = build_physics_adjacency()
    else:
        adjacency_raw, symbols = build_fully_connected_adjacency(
            len(PAPER_SENSOR_SYMBOLS)
        )
    adjacency_normalized = normalize_adjacency(
        adjacency_raw, method=config.adjacency_normalization
    )
    diagnostics["graph"] = adjacency_report(adjacency_raw, symbols)
    diagnostics["graph_kind"] = config.graph
    diagnostics["adjacency_normalization"] = config.adjacency_normalization

    diagnostics["splits"] = {
        "train_examples": len(train_windows),
        "validation_examples": len(validation_windows),
        "test_examples": len(test_windows),
        "train_engines": sorted(int(u) for u in np.unique(train_windows.units)),
        "validation_engines": sorted(int(u) for u in np.unique(validation_windows.units)),
    }

    logger.info(
        "Preprocessing complete: %d training, %d validation, %d test example(s).",
        len(train_windows), len(validation_windows), len(test_windows),
    )

    return PreparedData(
        config=config,
        train=train_windows,
        validation=validation_windows,
        test=test_windows,
        adjacency_raw=adjacency_raw,
        adjacency_normalized=adjacency_normalized,
        sensor_symbols=symbols,
        normalizer=normalizer,
        condition_encoder=encoder,
        diagnostics=diagnostics,
        raw=data if keep_frames else None,
        frames={"train": train, "test": test} if keep_frames else None,
    )
