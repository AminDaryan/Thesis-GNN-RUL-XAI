"""The whole network, built from the layers in layers.py.

ASSUMPTION A-11: the paper puts a small recurrent network in front of each
sensor. This file leaves that stage out, because the paper never states its
size, its depth, or how it turns a 30-cycle window into one number, so any
version of it would be invention rather than reproduction. Results from this
network therefore compare against the paper's own "ESTGNN w/o LSTM" row, never
its headline figures. Written up in docs/assumptions/A-11.md.

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
from .temporal import LastCycle, TemporalStage

__all__ = ["ESTGNN", "SPATIAL_STAGES"]

logger = get_logger(__name__)

# How wide each graph convolution layer is, from Table 4 of the paper.
GCN_HIDDEN_SIZES: tuple[int, int, int] = (32, 64, 32)

# How wide the first fully connected layer is, from Table 4.
FC_HIDDEN_SIZE: int = 128

# What may sit between the temporal stage and the dense layers: Table 4's graph
# layers ("gcn"), or nothing at all ("none", assumption A-42).
SPATIAL_STAGES: tuple[str, ...] = ("gcn", "none")


class ESTGNN(nn.Module):
    """Three graph convolutions, then two ordinary layers, as in Table 4.

    Pass the sensor graph through normalize_adjacency() before handing it here.
    A graph that has not been through that step is refused, because it lacks
    the self-connections that let a sensor see its own reading.
    """

    def __init__(
        self,
        adjacency: np.ndarray | Tensor | Sequence[Sequence[float]],
        *,
        temporal: TemporalStage | None = None,
        window_size: int = 1,
        rul_max: int = 125,
        gcn_hidden: Sequence[int] = GCN_HIDDEN_SIZES,
        fc_hidden: int = FC_HIDDEN_SIZE,
        gcn_bias: bool = False,
        spatial: str = "gcn",
        allow_time_in_the_graph_layer: bool = False,
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
                "been rescaled yet. Pass the output of normalize_adjacency(), "
                "which joins every sensor to itself — a self-loop — so that a "
                "sensor keeps its own reading when it mixes with its "
                "neighbours. Given the raw matrix, no sensor could see itself."
            )
        if rul_max < 1:
            raise ValueError(f"rul_max must be positive, got {rul_max}.")
        if spatial not in SPATIAL_STAGES:
            raise ValueError(
                f"spatial must be one of {list(SPATIAL_STAGES)}, got {spatial!r}."
            )
        if spatial == "gcn" and len(gcn_hidden) < 1:
            raise ValueError("at least one graph convolution layer is required.")
        if spatial == "none" and gcn_bias:
            raise ValueError(
                "gcn_bias gives the graph layers an offset, and with spatial='none' "
                "there are no graph layers to give one to."
            )

        # Kept even with spatial="none", so a stored network's saved weights,
        # which hold the graph, load back into it unchanged. With no graph
        # layers nothing in forward() reads it.
        self.register_buffer("adjacency", matrix)
        self.spatial = spatial
        self.n_nodes = int(matrix.shape[0])
        self.rul_max = int(rul_max)

        # The width of the first graph weight matrix is read off the STAGE, not
        # off the data. That is the whole point of the stage: with the width
        # taken from features.shape[2] there was a path from a preprocessing
        # setting to a 30-wide W(1), and nothing in between was named. See
        # A-07, and the refusal below.
        # Without a stage, the default is the A-07 stand-in over whatever
        # window the caller names. window_size=1 makes it the identity, so
        # ESTGNN(adjacency) alone is still Table 4 exactly.
        self.temporal = temporal if temporal is not None else LastCycle(window_size)
        self.in_features = int(self.temporal.out_features)

        # A stage that puts more than one number on a node WITHOUT learning
        # anything across the cycles is handing the graph layer raw cycles. It
        # cannot lie about this, because nothing is asked of it: the answer is
        # derived from whether it holds parameters.
        learns_across_time = any(p.numel() for p in self.temporal.parameters())
        self.time_folded_into_graph_layer = (
            self.in_features > 1 and not learns_across_time
        )
        if self.time_folded_into_graph_layer and not allow_time_in_the_graph_layer:
            # With no graph layers the cycles land in FC-1 instead, one set of
            # multipliers per sensor and cycle: still a temporal encoder that
            # is not named one, so the same flag is asked for (README.md).
            where = (
                f"the first graph weight matrix becomes ({self.in_features}, "
                f"{int(gcn_hidden[0])}). Those are multipliers learned over time "
                "and shared across sensors"
                if spatial == "gcn"
                else "with no graph layers the first fully connected weight matrix "
                f"becomes ({self.n_nodes * self.in_features}, {int(fc_hidden)}). "
                "Those are multipliers learned over time for every sensor"
            )
            raise ValueError(
                f"{type(self.temporal).__name__} puts {self.in_features} numbers "
                "on each sensor without learning anything across the cycles, so "
                f"{where} -- a temporal encoder, inside a layer "
                "that is not named one. That is the arrangement removed on "
                "2026-08-28; see A-07. To build it deliberately pass "
                "allow_time_in_the_graph_layer=True, which records the choice in "
                "the run."
            )

        # ASSUMPTION A-42: spatial="none" builds no graph layer and puts nothing
        # in their place, so the temporal stage's (B, 14, F) goes straight to
        # the flatten and FC-1. The paper's "ESTGNN w/o GCN" (Table 11) never
        # says what its layers were. Written up in docs/assumptions/A-42.md.
        widths = [self.in_features, *(int(w) for w in gcn_hidden)] if spatial == "gcn" else [self.in_features]
        self.graph_layers = nn.ModuleList(
            GraphConvolution(widths[i], widths[i + 1], bias=gcn_bias)
            for i in range(len(widths) - 1)
        )

        # Flatten turns (batch, nodes, last_width) into (batch, nodes*last_width).
        # For Table 4 that is 14 * 32 = 448; with no graph layers, 14 * F.
        self.flattened_size = self.n_nodes * widths[-1]
        self.fc1 = nn.Linear(self.flattened_size, int(fc_hidden))
        self.fc2 = nn.Linear(int(fc_hidden), 1)

        logger.info(
            "Built ESTGNN: %d node(s), %d feature(s) per node, graph layers %s, "
            "flatten to %d, dense %d -> 1. %d trainable parameters.",
            self.n_nodes,
            self.in_features,
            [layer.out_features for layer in self.graph_layers] or "none",
            self.flattened_size,
            int(fc_hidden),
            self.count_parameters(),
        )

    # -- forward -----------------------------------------------------------

    def forward(self, features: Tensor) -> Tensor:
        """Predict remaining life as a fraction between 0 and 1, not in cycles.

        Call predict_cycles() below if you want flight cycles.
        """
        hidden = self.temporal(features)
        for layer in self.graph_layers:
            hidden = torch.relu(layer(hidden, self.adjacency))

        flat = torch.flatten(hidden, start_dim=1)
        dense = torch.relu(self.fc1(flat))
        return torch.sigmoid(self.fc2(dense)).squeeze(-1)

    def predict_cycles(self, features: Tensor) -> Tensor:
        """Predict remaining life in flight cycles. Can never exceed rul_max."""
        return self.forward(features) * float(self.rul_max)

    # -- introspection -----------------------------------------------------

    @property
    def has_graph_layers(self) -> bool:
        """Whether any graph layer sits between the temporal stage and the dense layers."""
        return len(self.graph_layers) > 0

    def count_parameters(self, trainable_only: bool = True) -> int:
        """How many numbers the network learns. Compare it with Table 4."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def layer_summary(self) -> list[dict[str, object]]:
        """List every layer in the form of the paper's Table 4, the temporal stage folded into its Input row."""
        rows: list[dict[str, object]] = []
        width = self.in_features
        learned = sum(p.numel() for p in self.temporal.parameters())
        # The stage is named only when it learns numbers; README.md says why.
        source = ""
        if learned:
            kind = getattr(self.temporal, "cell_kind", None)
            source = str(kind).upper() if kind else type(self.temporal).__name__
        rows.append(
            {
                "layer": "Input",
                "type": "Node features",
                "from": source,
                "in": f"(B, {self.n_nodes}, {self.temporal.window_size})",
                "out": f"(B, {self.n_nodes}, {width})",
                "parameters": learned,
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
            f"nodes={self.n_nodes}, temporal={type(self.temporal).__name__}, "
            f"spatial={self.spatial}, in_features={self.in_features}, rul_max={self.rul_max}"
        )
