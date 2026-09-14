"""How far the prediction moves when one sensor at a time is swapped for its baseline. Not in the paper."""

from __future__ import annotations

import numpy as np
import torch

from ..models.estgnn import ESTGNN

__all__ = ["occlusion_values"]


@torch.no_grad()
def occlusion_values(
    model: ESTGNN,
    features: np.ndarray,
    baseline: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Prediction with every sensor, minus the prediction with one sensor replaced."""
    single = features.ndim == 2
    batch = features[None, ...] if single else features
    if batch.ndim != 3:
        raise ValueError(
            "features must be (nodes, window) or (examples, nodes, window), got "
            f"{features.shape}."
        )
    if baseline.shape != batch.shape[1:]:
        raise ValueError(
            f"baseline must be shaped {batch.shape[1:]} to match one example, "
            f"got {baseline.shape}."
        )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()

    n_examples, n_nodes, window = batch.shape
    results = np.empty((n_examples, n_nodes), dtype=np.float64)
    for start in range(0, n_examples, batch_size):
        chunk = batch[start : start + batch_size].astype(np.float32)
        # Row 0 of each example is the window as it is; row i + 1 is the same
        # window with sensor i's readings replaced by the baseline's.
        inputs = np.repeat(chunk[:, None, :, :], n_nodes + 1, axis=1)
        for node in range(n_nodes):
            inputs[:, node + 1, node, :] = baseline[node]
        flat = torch.from_numpy(inputs.reshape(-1, n_nodes, window)).to(device)
        predictions = model.predict_cycles(flat).cpu().numpy().astype(np.float64)
        predictions = predictions.reshape(len(chunk), n_nodes + 1)
        results[start : start + len(chunk)] = predictions[:, :1] - predictions[:, 1:]

    return results[0] if single else results
