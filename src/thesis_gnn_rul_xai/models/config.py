"""Every setting that decides the shape of the network, in one object a run can record.

Read README.md in this folder. It explains what every setting below means and
which of them the paper states.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
import yaml

from .estgnn import ESTGNN, FC_HIDDEN_SIZE, GCN_HIDDEN_SIZES, SPATIAL_STAGES
from .temporal import TemporalStageName, build_temporal_stage

__all__ = ["SPATIAL_STAGES", "ModelConfig", "build_model", "load_model_config"]


@dataclass
class ModelConfig:
    """The temporal stage and the graph layers, with Table 4's widths as defaults."""

    temporal: str = "last_cycle"
    # ASSUMPTION A-26: one recurrent layer of hidden size 32, projected to one
    # number per sensor. Section 2.2 gives the count of networks, never their size.
    temporal_hidden: int = 32
    temporal_out_features: int = 1
    # Section 2.2 gives every sensor its own network, so sharing one is not the paper.
    shared_across_sensors: bool = False
    gcn_hidden: tuple[int, ...] = GCN_HIDDEN_SIZES
    fc_hidden: int = FC_HIDDEN_SIZE
    # ASSUMPTION A-12: the graph layers carry no bias. Table 4 lists only W.
    gcn_bias: bool = False
    # ASSUMPTION A-42: "none" removes the graph layers and nothing replaces them;
    # the temporal stage's (B, 14, F) is flattened straight into FC-1. Section 4.3
    # names "ESTGNN w/o GCN" and never says what its layers were.
    spatial: str = "gcn"

    def __post_init__(self) -> None:
        """Check every value, so a bad setting fails before any training starts."""
        stages = get_args(TemporalStageName)
        if self.temporal not in stages:
            raise ValueError(
                f"temporal must be one of {list(stages)}, got {self.temporal!r}."
            )
        if self.spatial not in SPATIAL_STAGES:
            raise ValueError(
                f"spatial must be one of {list(SPATIAL_STAGES)}, got {self.spatial!r}."
            )
        self.gcn_hidden = tuple(int(width) for width in self.gcn_hidden)
        if not self.gcn_hidden or min(self.gcn_hidden) < 1:
            raise ValueError(
                f"gcn_hidden needs at least one positive width, got {self.gcn_hidden}."
            )
        for name in ("temporal_hidden", "temporal_out_features", "fc_hidden"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}.")
        # Two settings that would change the name and the digest of a network
        # they cannot change are refused, so one network never gets two names.
        if self.spatial == "none" and (self.gcn_hidden != GCN_HIDDEN_SIZES or self.gcn_bias):
            raise ValueError(
                "gcn_hidden and gcn_bias describe the graph layers, and with "
                "spatial='none' there are none; leave both at their defaults."
            )

    @property
    def recurrent(self) -> bool:
        """Whether the temporal stage has a hidden size at all."""
        return self.temporal in ("lstm", "gru")

    @property
    def has_graph_layers(self) -> bool:
        """Whether the network these settings describe has graph layers at all."""
        return self.spatial != "none"

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dictionary, ready to be saved; spatial is left out at its default."""
        settings = dataclasses.asdict(self)
        settings["gcn_hidden"] = list(self.gcn_hidden)
        # Left out at "gcn" so every record, digest and run name made before the
        # setting existed stays what it was; README.md says why.
        if self.spatial == "gcn":
            del settings["spatial"]
        return settings

    def digest(self) -> str:
        """Six hexadecimal characters that change whenever any setting does."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]

    def save(self, path: str | Path) -> Path:
        """Write the settings to a YAML file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


def load_model_config(path: str | Path) -> ModelConfig:
    """Read settings back from a YAML file. An unknown key is an error."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown model setting(s) in {path}: {sorted(unknown)}. "
            f"Valid keys are: {sorted(known)}."
        )
    return ModelConfig(**raw)


def build_model(
    adjacency: np.ndarray,
    config: ModelConfig,
    *,
    window_size: int,
    rul_max: int,
) -> ESTGNN:
    """Build the network these settings describe, on the graph given."""
    return ESTGNN(
        adjacency,
        temporal=build_temporal_stage(
            config.temporal,
            window_size=window_size,
            n_nodes=int(np.asarray(adjacency).shape[0]),
            hidden_size=config.temporal_hidden,
            out_features=config.temporal_out_features,
            shared_across_sensors=config.shared_across_sensors,
        ),
        rul_max=rul_max,
        gcn_hidden=config.gcn_hidden,
        fc_hidden=config.fc_hidden,
        gcn_bias=config.gcn_bias,
        spatial=config.spatial,
        # Asked for by name, so these are the stages the guard lets through.
        # ASSUMPTION A-28: the window summary hands the first graph layer three
        # numbers made from the cycles, so its weights mix time; named, not hidden.
        allow_time_in_the_graph_layer=config.temporal in ("raw_window", "window_summary"),
    )
