"""
The complete preprocessing pipeline
===================================

This module runs every preprocessing step in the correct order and returns
arrays that a model can consume directly. Having one function that performs the
whole sequence, rather than a notebook that calls the steps by hand, means the
experiment can be repeated exactly by anyone, including from a script or an
automated test.

Order of the steps
------------------
1. **Load** the three raw text files and attach a Remaining Useful Life value
   to every row.
2. **Diagnose** which sensors carry no information, and compare that with the
   selection made by Wen et al. (2026). This step changes nothing; it only
   reports.
3. **Select** the 14 sensors used by the paper, in the exact order of its
   Table 3, so that the sensor axis lines up with the sensor graph.
4. **Cap** the labels at the ceiling, producing the piecewise-linear target.
5. **Smooth** the sensor signals with an exponentially weighted moving average.
6. **Split** the training engines into a training part and a validation part.
7. **Identify operating conditions** and **rescale** the sensors, learning all
   statistics from the training engines only.
8. **Cut** the histories into fixed-length windows.
9. **Apply** the split decided in step 6 to the windows.

Why smoothing comes before rescaling
------------------------------------
Smoothing removes measurement noise and is most naturally expressed in the
sensor's own physical units, where the noise actually occurs. Rescaling is the
final preparation before the model and should therefore see the signal the
model will use.

This order is **taken from the paper**, not chosen here: Fig. 1 of Wen et al.
(2026) shows the preprocessing chain as *feature selection, data smoothing,
piece-wise RUL, data normalisation, train and test datasets*. See A-08 in
``docs/ASSUMPTIONS.md``, and the caution recorded there about what this order
does on the multi-condition subsets FD002 and FD004.

Why the split comes before the scaling
--------------------------------------
The validation engines are chosen in step 6, before any statistic is computed
in step 7. Fitting the scaler on all training engines and only afterwards
holding some of them out would let a held-out engine define a minimum or a
maximum, which is the same kind of leak that keeping the test set separate is
meant to prevent (Kapoor & Narayanan, 2023). The effect is smaller than test
leakage, but it biases exactly the quantity a validation set exists to measure.

Where the training and test paths differ, and where they must not
-----------------------------------------------------------------
Both splits pass through the same code. The only differences are deliberate
and are exactly two:

* All statistics -- cluster centres, minima, maxima, means, standard
  deviations -- are learned from the training engines and then applied to the
  validation and test engines unchanged.
* Training engines produce every window; test engines produce only their final
  window, because that is the single moment the official labels describe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PreprocessingConfig
from ..graph.adjacency import (
    adjacency_report,
    build_physics_adjacency,
    normalize_adjacency,
)
from ..utils.logging_setup import get_logger
from ..utils.paths import processed_data_dir, raw_data_dir
from .labeling import apply_piecewise_rul
from .loading import CMAPSSData, load_subset
from .normalization import OperatingConditionEncoder, SensorNormalizer
from .sensors import (
    PAPER_SENSOR_COLUMNS,
    PAPER_SENSOR_SYMBOLS,
    UNIT_COL,
    detect_uninformative_sensors,
)
from .smoothing import apply_ewma
from .windowing import (
    WindowedDataset,
    make_last_windows,
    make_windows,
    split_units_by_id,
)

__all__ = ["PreparedData", "run_preprocessing"]

logger = get_logger(__name__)


@dataclass
class PreparedData:
    """Everything the modelling stage needs, and everything needed to audit it.

    Attributes
    ----------
    config : PreprocessingConfig
        The exact settings that produced this object.
    train, validation, test : WindowedDataset
        The three sets of windowed examples. ``validation`` may be empty if the
        configured fraction is zero.
    adjacency_raw, adjacency_normalized : numpy.ndarray
        The sensor graph before and after the graph-convolution normalisation.
    sensor_symbols : tuple of str
        Sensor names in node order.
    normalizer : SensorNormalizer
        The fitted scaler, kept so its statistics can be inspected and saved.
    condition_encoder : OperatingConditionEncoder
        The fitted operating-condition model.
    diagnostics : dict
        Free-form record of what the pipeline observed: constant sensors,
        graph properties, split sizes, and whether the paper's sensor selection
        agrees with the data.
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
        """Return a small table of the three splits, for display."""
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
        """Write the prepared arrays, the graph, the settings and the report.

        Five files are written, so that a later stage never has to re-run the
        preprocessing and so that the exact inputs of any experiment can be
        recovered:

        ``windows.npz``
            The three splits as arrays.
        ``adjacency.npz``
            The raw and normalised sensor graphs, plus the node labels.
        ``normalization_statistics.csv``
            The learned scaling statistics, one row per sensor and condition.
        ``config.yaml``
            The settings used.
        ``diagnostics.json``
            What the pipeline observed while running.
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
    """Run the whole preprocessing pipeline and return the result.

    Parameters
    ----------
    config : PreprocessingConfig, optional
        Settings to use. Defaults to :class:`PreprocessingConfig`, which is
        configured for FD001.
    data_dir : str or Path, optional
        Directory holding the three C-MAPSS text files for this
        sub-dataset. Defaults to ``data/raw``.
    keep_frames : bool
        When ``True``, the intermediate data frames are kept on the returned
        object. This costs memory but is useful in a notebook, where the tables
        and plots are the point.

    Returns
    -------
    PreparedData
        Windowed arrays, the sensor graph, the fitted transformers, and a
        diagnostics record.
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

    # -- 3 & 4. select sensors and cap labels -----------------------------
    # Fig. 1 of the paper puts the piecewise RUL *after* the smoothing; here
    # it comes before. The two commute exactly: capping touches only the
    # label column and smoothing touches only the sensor columns, so no row
    # is read by both. The order below groups the label work together.
    columns = list(PAPER_SENSOR_COLUMNS)
    train = apply_piecewise_rul(data.train, rul_max=config.rul_max)
    test = apply_piecewise_rul(
        data.test, rul_max=config.rul_max
    ) if config.clip_test_labels else data.test.copy()

    # -- 5. smooth --------------------------------------------------------
    train = apply_ewma(train, columns, alpha=config.smoothing_alpha)
    test = apply_ewma(test, columns, alpha=config.smoothing_alpha)

    # -- 6. hold out the validation engines -------------------------------
    # This happens BEFORE any statistic is learned. If the split were made
    # later, the minima, maxima and cluster centres would be computed over
    # engines that are about to be held out, and the validation error would
    # be optimistic for exactly the reason the test set is kept separate.
    train_units, validation_units = split_units_by_id(
        np.unique(train[UNIT_COL].to_numpy()),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    # -- 7. operating conditions and rescaling ----------------------------
    # Fitted on the training engines only; applied unchanged to the
    # validation and test engines.
    training_rows = train[UNIT_COL].isin(train_units)
    encoder = OperatingConditionEncoder(seed=config.seed).fit(train[training_rows])
    train = encoder.transform(train)
    test = encoder.transform(test)
    diagnostics["n_operating_conditions"] = encoder.n_detected

    # Recomputed rather than reused: `train` was rebound above, so a mask
    # captured before the rebind would be indexing a different frame.
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
        node_feature_mode=config.node_feature_mode,
    )
    test_windows = make_last_windows(
        test,
        columns,
        window_size=config.window_size,
        node_feature_mode=config.node_feature_mode,
    )

    # -- 9. apply the split decided in step 6 -----------------------------
    validation_windows = train_windows.subset_by_units(validation_units)
    train_windows = train_windows.subset_by_units(train_units)

    overlap = set(train_windows.units.tolist()) & set(validation_windows.units.tolist())
    if overlap:  # pragma: no cover - defensive
        raise AssertionError(
            f"Engines {sorted(overlap)} appear in both the training and the "
            "validation split. This would invalidate the experiment."
        )

    # -- the sensor graph -------------------------------------------------
    adjacency_raw, symbols = build_physics_adjacency()
    adjacency_normalized = normalize_adjacency(adjacency_raw)
    diagnostics["graph"] = adjacency_report(adjacency_raw, symbols)

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
