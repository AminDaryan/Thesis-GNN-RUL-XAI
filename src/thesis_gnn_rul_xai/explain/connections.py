"""How far the prediction moves when one connection of the sensor graph is removed and the graph rescaled. Not in the paper.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from ..graph.adjacency import (
    AdjacencyNormalization,
    connections_of,
    normalize_adjacency,
)
from ..models.estgnn import ESTGNN

__all__ = ["connection_removal", "connections_of"]


@torch.no_grad()
def connection_removal(
    model: ESTGNN,
    features: np.ndarray,
    raw_adjacency: np.ndarray,
    symbols: Sequence[str],
    *,
    method: AdjacencyNormalization,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Prediction with the graph as trained, minus the prediction with one connection removed, in cycles."""
    raw = np.asarray(raw_adjacency, dtype=np.float64)
    if len(symbols) != raw.shape[0]:
        raise ValueError(
            f"the graph has {raw.shape[0]} node(s) but {len(symbols)} symbol(s) were given."
        )
    trained = model.adjacency.detach().cpu().numpy()
    rebuilt = normalize_adjacency(raw, method=method)
    if rebuilt.shape != trained.shape or not np.allclose(rebuilt, trained, atol=1e-6):
        raise ValueError(
            "The raw graph rescaled by the method given is not the graph this network "
            "was trained on. Pass the run's own raw graph and its own rescaling method, "
            "or every removal would be measured against the wrong graph."
        )
    single = features.ndim == 2
    batch = features[None, ...] if single else features

    device = next(model.parameters()).device
    network = copy.deepcopy(model).eval()

    def predict() -> np.ndarray:
        out = []
        for start in range(0, len(batch), batch_size):
            chunk = torch.from_numpy(batch[start : start + batch_size].astype(np.float32))
            out.append(network.predict_cycles(chunk.to(device)).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    as_trained = predict()
    columns: dict[str, np.ndarray] = {}
    for i, j in connections_of(raw):
        # ASSUMPTION A-30: removing a connection deletes it from the raw graph in
        # both directions and rescales the whole graph again, the way the network's
        # graph was built, rather than zeroing one entry of the rescaled graph. The
        # rest of the two sensors' rows then grows to fill the gap. The network is
        # not retrained. Written up in docs/ASSUMPTIONS.md.
        removed = raw.copy()
        removed[i, j] = removed[j, i] = 0.0
        network.adjacency.copy_(
            torch.from_numpy(normalize_adjacency(removed, method=method)).to(device)
        )
        columns[f"{symbols[i]}-{symbols[j]}"] = as_trained - predict()
    return pd.DataFrame(columns)
