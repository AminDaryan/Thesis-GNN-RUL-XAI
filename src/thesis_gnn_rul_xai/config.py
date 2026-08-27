"""
Configuration
=============

Every number that controls the behaviour of the pipeline is collected here
instead of being written directly into the code. There are two reasons for
this, both of them about scientific honesty:

1. **Repeatability.** A single configuration file fully describes an
   experiment. Saving that file next to the results means anyone can see
   exactly which settings produced them.
2. **Visibility of assumptions.** The source paper does not state every
   setting. When a value had to be chosen by us, the choice appears here as a
   named field with a comment, rather than being hidden somewhere inside a
   function. Fields whose value is *not* taken from the paper are marked
   ``ASSUMPTION`` in their description.

The configuration is a plain Python dataclass, and can also be read from a YAML
file so that an experiment can be changed without editing any code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "NormalizationMethod",
    "NodeFeatureMode",
    "PreprocessingConfig",
    "load_config",
]

#: How sensor values are rescaled before they reach the model.
#:
#: ``"minmax_per_condition"``
#:     Rescale each sensor to the range [0, 1] separately for each operating
#:     condition. On datasets with several operating conditions (FD002, FD004)
#:     this removes the large jumps caused by the aircraft flying at different
#:     altitudes and speeds, which would otherwise hide the much smaller
#:     changes caused by engine wear.
#: ``"minmax_global"``
#:     Rescale each sensor to [0, 1] using one minimum and one maximum for the
#:     whole dataset. On FD001 there is only one operating condition, so this
#:     gives the same result as the per-condition variant.
#: ``"zscore"``
#:     Subtract the mean and divide by the standard deviation, so each sensor
#:     has mean 0 and standard deviation 1. Included so that the sensitivity of
#:     the results to this choice can be measured.
NormalizationMethod = Literal["minmax_per_condition", "minmax_global", "zscore"]

#: What each node of the sensor graph carries as its feature vector.
#:
#: ``"window"``
#:     The whole time window for that sensor, so a node feature is a vector of
#:     length equal to the window size. This is the natural choice when no
#:     temporal encoder (such as an LSTM) is used, because the graph layer then
#:     sees the full recent history.
#: ``"last_step"``
#:     Only the most recent value in the window, so a node feature is a single
#:     number. This reproduces the input shape printed in Table 4 of Wen et al.
#:     (2026), which is ``[batch, 14, 1]``.
NodeFeatureMode = Literal["window", "last_step"]


@dataclass
class PreprocessingConfig:
    """All settings used to turn raw C-MAPSS files into model-ready arrays.

    Attributes
    ----------
    subset : str
        Which of the four C-MAPSS sub-datasets to use. This project starts with
        ``"FD001"``: one operating condition and one fault mode, which is the
        simplest case and therefore the right place to establish a correct
        pipeline.
    window_size : int
        Number of consecutive flight cycles in one training example. Taken from
        Table 6 of Wen et al. (2026), which gives 30 time steps.
    window_stride : int
        Step between the start of one window and the start of the next, in
        cycles. A stride of 1 produces the largest number of training examples.
        ASSUMPTION: not stated in the paper; 1 is the usual choice on C-MAPSS.
    rul_max : int
        Upper limit applied to the Remaining Useful Life label, in cycles. See
        :mod:`thesis_gnn_rul_xai.data.labeling` for why this limit exists.
        ASSUMPTION: the paper defines the limit symbolically in its Equation 19
        but never prints its value. 125 is inferred from Section 4.2 of the
        paper, which describes predictions holding "around 125" while an engine
        is healthy, and it is the value used by the closely related work of Jin
        et al. (2025) on the same dataset.
    clip_test_labels : bool
        Whether the same upper limit is applied to the test labels. This must
        be ``True`` for the evaluation to be meaningful with the architecture
        of Wen et al. (2026): their output layer is a sigmoid function, so the
        model can never predict a value above ``rul_max``. Leaving the test
        labels uncapped would charge the model for an error it is structurally
        unable to avoid.
    normalization : NormalizationMethod
        Which rescaling to apply. ASSUMPTION: the paper never describes its
        input scaling. The default follows Jin et al. (2025), who use
        per-condition min-max scaling on the same dataset.
    smoothing_alpha : float or None
        Strength of the exponentially weighted moving average filter, between 0
        and 1. Larger values follow the raw signal more closely; smaller values
        smooth more strongly. ``None`` disables smoothing entirely.
        ASSUMPTION: Section 3.2.3 of the paper is self-contradictory here. It
        writes that "S(t-1) ... is equal to 0.5 in this study", attaching the
        number 0.5 to the previous smoothed value rather than to the smoothing
        parameter. Reading it as the smoothing parameter is the only
        interpretation that makes the sentence meaningful, so 0.5 is used.
    node_feature_mode : NodeFeatureMode
        Shape of the node features. See :data:`NodeFeatureMode`.
    validation_fraction : float
        Fraction of *engine units* held out for validation. Splitting by engine
        rather than by window is essential; see
        :mod:`thesis_gnn_rul_xai.data.windowing` for the explanation.
        ASSUMPTION: the paper describes no validation split at all.
    seed : int
        Random seed controlling the validation split.
    """

    subset: str = "FD001"

    # --- Windowing ---------------------------------------------------------
    window_size: int = 30
    # ASSUMPTION A-05: the paper gives a 30-step window in Table 6 but never
    # states the stride between consecutive windows.
    window_stride: int = 1

    # --- Labels ------------------------------------------------------------
    # ASSUMPTION A-01: Equation 19 defines the ceiling symbolically and the
    # value is never printed. 125 is read off the True RUL curve of Fig. 7.
    rul_max: int = 125
    clip_test_labels: bool = True

    # --- Signal processing -------------------------------------------------
    # ASSUMPTION A-03: the paper never describes its input scaling at all.
    # This follows Jin et al. (2025) on the same dataset.
    normalization: NormalizationMethod = "minmax_per_condition"
    # ASSUMPTION A-04: Section 3.2.3 attaches the value 0.5 to the previous
    # smoothed value rather than to the smoothing parameter. Read as alpha.
    smoothing_alpha: float | None = 0.5

    # --- Model input shape -------------------------------------------------
    node_feature_mode: NodeFeatureMode = "window"

    # --- Data splitting ----------------------------------------------------
    validation_fraction: float = 0.2
    seed: int = 42

    # --- Bookkeeping -------------------------------------------------------
    notes: str = field(
        default=(
            "Preprocessing configuration for the FD001 subset. Fields marked "
            "ASSUMPTION in the class documentation are not specified by "
            "Wen et al. (2026); see docs/ASSUMPTIONS.md."
        )
    )

    def __post_init__(self) -> None:
        """Check that the settings are internally consistent.

        Catching an impossible configuration here, before any data is touched,
        gives a clear error message instead of a confusing failure deep inside
        the pipeline.
        """
        valid_subsets = {"FD001", "FD002", "FD003", "FD004"}
        if self.subset not in valid_subsets:
            raise ValueError(
                f"subset must be one of {sorted(valid_subsets)}, got {self.subset!r}."
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {self.window_size}.")
        if self.window_stride < 1:
            raise ValueError(
                f"window_stride must be at least 1, got {self.window_stride}."
            )
        if self.rul_max < 1:
            raise ValueError(f"rul_max must be positive, got {self.rul_max}.")
        if self.smoothing_alpha is not None and not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError(
                "smoothing_alpha must lie in the interval (0, 1], or be None to "
                f"disable smoothing. Got {self.smoothing_alpha}."
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be in [0, 1), got "
                f"{self.validation_fraction}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary, ready to be saved."""
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write the configuration to a YAML file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path


def load_config(path: str | Path) -> PreprocessingConfig:
    """Read a :class:`PreprocessingConfig` from a YAML file.

    Parameters
    ----------
    path : str or Path
        Location of the YAML file.

    Returns
    -------
    PreprocessingConfig
        The parsed configuration, with all consistency checks applied.

    Raises
    ------
    ValueError
        If the file contains a key that is not a field of the configuration.
        This is deliberately strict: a silently ignored typo in a settings file
        is a common and hard-to-find source of wrong results.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    known = {f.name for f in dataclasses.fields(PreprocessingConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) in {path}: {sorted(unknown)}. "
            f"Valid keys are: {sorted(known)}."
        )
    return PreprocessingConfig(**raw)
