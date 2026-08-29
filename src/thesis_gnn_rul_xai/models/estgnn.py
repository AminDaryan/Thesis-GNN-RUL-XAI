"""The whole network, built from the layers in layers.py.

ASSUMPTION A-11: the paper puts a small recurrent network in front of each
sensor. This file leaves that stage out, because the paper never states its
size, its depth, or how it turns a 30-cycle window into one number, so any
version of it would be invention rather than reproduction. Results from this
network therefore compare against the paper's own "ESTGNN w/o LSTM" row, never
its headline figures. Written up in docs/ASSUMPTIONS.md.

Read README.md in this folder. It explains everything this file does, what
these results may be compared with, and what the last layer rules out.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..utils.logging_setup import get_logger
from .layers import GraphConvolution

__all__ = ["ESTGNN"]

logger = get_logger(__name__)

# How wide each graph convolution layer is, from Table 4 of the paper.
GCN_HIDDEN_SIZES: tuple[int, int, int] = (32, 64, 32)

# How wide the first fully connected layer is, from Table 4.
FC_HIDDEN_SIZE: int = 128


class ESTGNN(nn.Module):
    """Three graph convolutions, then two ordinary layers, as in Table 4.

    Pass the adjacency table through normalize_adjacency() before handing it
    here; an unrescaled table is rejected.
    """

    def __init__(
        self,
        adjacency: np.ndarray | Tensor | Sequence[Sequence[float]],
        *,
        in_features: int = 1,
        rul_max: int = 125,
        gcn_hidden: Sequence[int] = GCN_HIDDEN_SIZES,
        fc_hidden: int = FC_HIDDEN_SIZE,
        gcn_bias: bool = False,
    ) -> None:
        super().__init__()

        matrix = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"adjacency must be a square matrix, got {tuple(matrix.shape)}."
            )
        if bool((torch.diagonal(matrix) == 0).any()):
            raise ValueError(
                "The adjacency matrix has a zero on its diagonal, so it has not "
                "been normalised. Pass the output of normalize_adjacency(), "
                "which adds the self-loops the graph convolution needs; giving "
                "the raw matrix would stop every node from seeing its own value."
            )
        if in_features < 1:
            raise ValueError(f"in_features must be at least 1, got {in_features}.")
        if rul_max < 1:
            raise ValueError(f"rul_max must be positive, got {rul_max}.")
        if len(gcn_hidden) < 1:
            raise ValueError("at least one graph convolution layer is required.")

        self.register_buffer("adjacency", matrix)
        self.n_nodes = int(matrix.shape[0])
        self.in_features = int(in_features)
        self.rul_max = int(rul_max)

        widths = [self.in_features, *(int(w) for w in gcn_hidden)]
        self.graph_layers = nn.ModuleList(
            GraphConvolution(widths[i], widths[i + 1], bias=gcn_bias)
            for i in range(len(widths) - 1)
        )

        # Flatten turns (batch, nodes, last_width) into (batch, nodes*last_width).
        # For Table 4 that is 14 * 32 = 448.
        self.flattened_size = self.n_nodes * widths[-1]
        self.fc1 = nn.Linear(self.flattened_size, int(fc_hidden))
        self.fc2 = nn.Linear(int(fc_hidden), 1)

        logger.info(
            "Built ESTGNN: %d node(s), %d feature(s) per node, graph layers %s, "
            "flatten to %d, dense %d -> 1. %d trainable parameters.",
            self.n_nodes,
            self.in_features,
            list(gcn_hidden),
            self.flattened_size,
            int(fc_hidden),
            self.count_parameters(),
        )

    # -- forward -----------------------------------------------------------

    def forward(self, features: Tensor) -> Tensor:
        """Predict remaining life as a fraction between 0 and 1, not in cycles.

        Call predict_cycles() below if you want flight cycles.
        """
        hidden = features
        for layer in self.graph_layers:
            hidden = torch.relu(layer(hidden, self.adjacency))

        flat = torch.flatten(hidden, start_dim=1)
        dense = torch.relu(self.fc1(flat))
        return torch.sigmoid(self.fc2(dense)).squeeze(-1)

    def predict_cycles(self, features: Tensor) -> Tensor:
        """Predict remaining life in flight cycles. Can never exceed rul_max."""
        return self.forward(features) * float(self.rul_max)

    # -- introspection -----------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        """How many numbers the network learns. Compare it with Table 4."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def layer_summary(self) -> list[dict[str, object]]:
        """List every layer in the same form as the paper's Table 4."""
        rows: list[dict[str, object]] = []
        width = self.in_features
        rows.append(
            {
                "layer": "Input",
                "type": "Node features",
                "in": f"(B, {self.n_nodes}, {width})",
                "out": f"(B, {self.n_nodes}, {width})",
                "parameters": 0,
                "activation": "-",
            }
        )
        for index, layer in enumerate(self.graph_layers, start=1):
            rows.append(
                {
                    "layer": f"GCN-{index}",
                    "type": "GCN",
                    "in": f"(B, {self.n_nodes}, {layer.in_features})",
                    "out": f"(B, {self.n_nodes}, {layer.out_features})",
                    "parameters": sum(p.numel() for p in layer.parameters()),
                    "activation": "ReLU",
                }
            )
            width = layer.out_features
        rows.append(
            {
                "layer": "Flatten",
                "type": "Tensor reshape",
                "in": f"(B, {self.n_nodes}, {width})",
                "out": f"(B, {self.flattened_size})",
                "parameters": 0,
                "activation": "-",
            }
        )
        rows.append(
            {
                "layer": "FC-1",
                "type": "Fully connected",
                "in": f"(B, {self.fc1.in_features})",
                "out": f"(B, {self.fc1.out_features})",
                "parameters": sum(p.numel() for p in self.fc1.parameters()),
                "activation": "ReLU",
            }
        )
        rows.append(
            {
                "layer": "FC-2",
                "type": "Fully connected",
                "in": f"(B, {self.fc2.in_features})",
                "out": f"(B, {self.fc2.out_features})",
                "parameters": sum(p.numel() for p in self.fc2.parameters()),
                "activation": "Sigmoid",
            }
        )
        return rows

    def extra_repr(self) -> str:  # pragma: no cover - display only
        return (
            f"nodes={self.n_nodes}, in_features={self.in_features}, "
            f"rul_max={self.rul_max}"
        )
