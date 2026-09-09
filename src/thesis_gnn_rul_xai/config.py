"""Every setting that controls the preparation of the data, in one object.

It can be saved to and read back from a YAML file, so an experiment can be
changed without editing any code.

Read README.md in this folder. It explains what every setting below means,
what the defaults do, and which of them will catch you out.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "AdjacencyNormalization",
    "NormalizationMethod",
    "PreprocessingConfig",
    "SensorGraph",
    "load_config",
]

# The three ways of putting the sensors on a comparable scale. README.md.
NormalizationMethod = Literal["minmax_per_condition", "minmax_global", "zscore"]

# Which sensor graph to give the network: the paper's own, or one where every
# sensor is connected to every other. ASSUMPTION A-19 covers the second, which
# stands in for an experiment the paper names but never describes.
SensorGraph = Literal["physics", "complete"]

# The two ways of rescaling that graph before the network uses it.
# ASSUMPTION A-21 covers the choice. Applied by normalize_adjacency() in
# graph/adjacency.py.
AdjacencyNormalization = Literal["symmetric", "row"]


@dataclass
class PreprocessingConfig:
    """Every setting used to turn the raw NASA files into arrays for the model.

    README.md in this folder explains each field below, one at a time.
    """

    subset: str = "FD001"

    # --- Cutting the histories into examples --------------------------------
    window_size: int = 30
    # ASSUMPTION A-05: move the window on by one cycle at a time. The paper
    # gives the window length in Table 6 but never says how far it moves.
    # Written up in docs/ASSUMPTIONS.md.
    window_stride: int = 1

    # --- Labels ------------------------------------------------------------
    # ASSUMPTION A-01: cap the label at 125 flight cycles. The paper's
    # Equation 19 says there is a cap but never prints the number; 125 was
    # measured off its Fig. 7. Written up in docs/ASSUMPTIONS.md.
    rul_max: int = 125
    # ASSUMPTION A-02: put the same ceiling on the test labels. The paper does
    # not say whether it does. The network's last layer cannot produce a number
    # above the ceiling, so leaving the test labels uncapped would charge it for
    # an error it is built to be unable to avoid. This can only move our
    # reported numbers in our favour, so it must be stated whenever they are
    # set beside published ones. Written up in docs/ASSUMPTIONS.md.
    clip_test_labels: bool = True

    # --- Preparing the sensor readings --------------------------------------
    # ASSUMPTION A-03: put the sensors on a comparable scale with z-scores. The
    # paper never describes this step at all. Worth 3.83 cycles against the
    # alternative, which is the largest measured effect of any choice in this
    # project. Written up in docs/ASSUMPTIONS.md as A-03, A-22 and F-10.
    normalization: NormalizationMethod = "zscore"
    # ASSUMPTION A-04: smooth with a strength of 0.5. The paper's Section 3.2.3
    # attaches 0.5 to the wrong symbol, and this is the only reading that makes
    # the sentence mean anything. Written up in docs/ASSUMPTIONS.md.
    smoothing_alpha: float | None = 0.5

    # --- The sensor graph --------------------------------------------------
    # ASSUMPTION A-19: when repeating the paper's experiment "without physics
    # prior", stand in a graph where every sensor is joined to every other. The
    # paper names that experiment but never says what it used. Written up in
    # docs/ASSUMPTIONS.md. Leave this on "physics" for the paper's own graph.
    graph: SensorGraph = "physics"
    # ASSUMPTION A-21: rescale the graph by rows. The paper never prints the
    # formula and two forms fit what it says. With nothing held back --
    # the default until 2026-09-02, and no longer what this file sets --
    # rows measured 0.35 cycles closer than the alternative over five seeds
    # (paired p = 0.047). That advantage does not survive holding engines out,
    # so it is a fact about this configuration only. Written up as A-21 in
    # docs/ASSUMPTIONS.md, with the table of all four comparisons.
    adjacency_normalization: AdjacencyNormalization = "row"

    # --- Holding engines back ------------------------------------------------
    # ASSUMPTION A-06: hold back 20 % of the training engines, chosen by engine
    # and never by window. The paper describes no held-out set and trains a
    # fixed 200 epochs, so this is an addition rather than a reading of it: a
    # run now restores the weights that scored best on the held-back engines
    # and may stop early. Reinstated as the default on 2026-09-08, reversing
    # the 2026-08-29 decision that removed it; the reversal and what it cost
    # are written up under A-06 in docs/ASSUMPTIONS.md.
    validation_fraction: float = 0.2
    seed: int = 42

    # --- Bookkeeping -------------------------------------------------------
    notes: str = field(
        default=(
            "Settings for preparing the FD001 subset. The lines marked "
            "ASSUMPTION in config.py are choices we made because Wen et al. "
            "(2026) does not state them; see docs/ASSUMPTIONS.md."
        )
    )

    def __post_init__(self) -> None:
        """Reject any setting that is out of range or misspelled."""
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
        if self.graph not in ("physics", "complete"):
            raise ValueError(
                f"graph must be 'physics' or 'complete', got {self.graph!r}."
            )
        if self.adjacency_normalization not in ("symmetric", "row"):
            raise ValueError(
                "adjacency_normalization must be 'symmetric' or 'row', got "
                f"{self.adjacency_normalization!r}."
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
    """Read settings back from a YAML file. An unknown key is an error."""
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
