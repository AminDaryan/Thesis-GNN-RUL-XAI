"""
The spatiotemporal graph network (ESTGNN, without the LSTM stage)
=================================================================

What this module builds
-----------------------
The network printed as Table 4 of Wen et al. (2026): three graph convolution
layers, a flattening step, and two fully connected layers ending in a sigmoid.
This is the part of their architecture that the paper specifies completely, so
no value in the layer sizes below is a guess.

===========  ==================================  ==========================
Layer        Shape in  ->  shape out             Activation
===========  ==================================  ==========================
GCN-1        ``(B, 14, F)``  ->  ``(B, 14, 32)``  ReLU
GCN-2        ``(B, 14, 32)`` ->  ``(B, 14, 64)``  ReLU
GCN-3        ``(B, 14, 64)`` ->  ``(B, 14, 32)``  ReLU
Flatten      ``(B, 14, 32)`` ->  ``(B, 448)``     -
FC-1         ``(B, 448)``    ->  ``(B, 128)``     ReLU
FC-2         ``(B, 128)``    ->  ``(B, 1)``       Sigmoid
===========  ==================================  ==========================

What ``F`` is, and why it is not 1 here
---------------------------------------
Table 4 gives the input as ``[batch, 14, 1]``: one number per sensor. That
number is the output of a per-sensor LSTM which the paper never specifies (its
hidden size, its depth, and how it reduces a 30-cycle window to a single value
are all absent). This project omits that stage rather than invent it; see
assumptions A-07 and A-11.

With no temporal encoder, each sensor node instead carries its whole 30-cycle
window, so ``F`` is 30 and the first weight matrix is ``(30, 32)`` rather than
``(1, 32)``. Everything after the first layer is unchanged, because the first
layer maps to 32 features either way.

Setting ``in_features=1`` reproduces Table 4's printed shapes exactly, and is
what the model would need if the LSTM stage were added later. Both are
supported; the default follows the configuration this project actually runs.

**Results from this network must be compared against the paper's own
"ESTGNN w/o LSTM" ablation, which is 14.93 RMSE on FD001 (their Table 11), not
against the 11.78 of the full model.**

What the sigmoid means
----------------------
The last layer produces a number between 0 and 1. The paper describes this as a
"normalized RUL value", so the prediction in flight cycles is that number
multiplied by the label ceiling ``RUL_max``. Two consequences follow, and both
matter:

* The model **cannot** predict a remaining life above the ceiling, whatever the
  input. This is the reason assumption A-02 caps the test labels as well: not
  doing so would charge the model for an error its own architecture forbids it
  from avoiding.
* Training targets must be divided by the same ceiling, so that they lie in
  ``[0, 1]`` where the sigmoid can reach them. :meth:`ESTGNN.forward` returns
  the normalised value, and :meth:`ESTGNN.predict_cycles` converts to cycles.

Which graph the network uses
----------------------------
The adjacency matrix is supplied when the model is built and stored as a fixed
buffer, not as a parameter: it is engineering knowledge, not something learned.
Passing a different matrix is all that is needed for the paper's "without
physics prior" ablation --- see
:func:`thesis_gnn_rul_xai.graph.adjacency.build_fully_connected_adjacency`.

References
----------
Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
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

#: Output width of each graph convolution layer, from Table 4 of the paper.
GCN_HIDDEN_SIZES: tuple[int, int, int] = (32, 64, 32)

#: Width of the first fully connected layer, from Table 4.
FC_HIDDEN_SIZE: int = 128


class ESTGNN(nn.Module):
    """Three graph convolutions and two dense layers, as in Table 4.

    Parameters
    ----------
    adjacency : array-like
        The **normalised** adjacency matrix, shaped ``(nodes, nodes)``, as
        returned by
        :func:`thesis_gnn_rul_xai.graph.adjacency.normalize_adjacency`. Passing
        the raw matrix instead is a silent error that degrades training, so the
        constructor checks for the self-loops that normalisation adds and
        refuses a matrix with a zero diagonal.
    in_features : int
        Numbers carried by each sensor node. 30 for a whole window (this
        project's default, matching ``node_feature_mode="window"``), or 1 to
        reproduce the printed shapes of Table 4.
    rul_max : int
        The label ceiling, used to convert the sigmoid output into cycles.
    gcn_hidden : sequence of int
        Output width of each graph convolution. The default is Table 4's
        ``(32, 64, 32)``.
    fc_hidden : int
        Width of the first fully connected layer. Table 4 gives 128.
    gcn_bias : bool
        Whether the graph convolutions carry a bias. Equations 8 to 10 show
        none, so this defaults to ``False``.

    Attributes
    ----------
    adjacency : torch.Tensor
        Registered as a buffer, so it moves with ``.to(device)`` and is saved
        in the state dictionary, but is never updated by the optimiser.

    Examples
    --------
    Build the network on the paper's own sensor graph and push one batch of
    four examples through it:

    >>> import numpy as np, torch
    >>> from thesis_gnn_rul_xai.graph.adjacency import (
    ...     build_physics_adjacency, normalize_adjacency)
    >>> raw, symbols = build_physics_adjacency()
    >>> model = ESTGNN(normalize_adjacency(raw), in_features=30)
    >>> batch = torch.zeros(4, 14, 30)
    >>> model(batch).shape
    torch.Size([4])

    With ``in_features=1`` the parameter count matches Table 4 exactly:

    >>> table4 = ESTGNN(normalize_adjacency(raw), in_features=1)
    >>> sum(p.numel() for p in table4.parameters())
    61729
    """

    def __init__(
        self,
        adjacency: np.ndarray | Tensor | Sequence[Sequence[float]],
        *,
        in_features: int = 30,
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
        """Predict the **normalised** remaining life for a batch.

        Parameters
        ----------
        features : torch.Tensor
            Shaped ``(batch, nodes, in_features)``.

        Returns
        -------
        torch.Tensor
            Shaped ``(batch,)``, every element in ``[0, 1]``. Multiply by
            :attr:`rul_max`, or call :meth:`predict_cycles`, to get flight
            cycles.
        """
        hidden = features
        for layer in self.graph_layers:
            hidden = torch.relu(layer(hidden, self.adjacency))

        flat = torch.flatten(hidden, start_dim=1)
        dense = torch.relu(self.fc1(flat))
        return torch.sigmoid(self.fc2(dense)).squeeze(-1)

    def predict_cycles(self, features: Tensor) -> Tensor:
        """Predict remaining life in flight cycles.

        This is :meth:`forward` scaled by the ceiling. It is the quantity the
        evaluation metrics compare against the official labels, and it can
        never exceed :attr:`rul_max` --- see the module docstring and
        assumption A-02.
        """
        return self.forward(features) * float(self.rul_max)

    # -- introspection -----------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Number of parameters, for checking the build against Table 4."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def layer_summary(self) -> list[dict[str, object]]:
        """Describe each layer, in the same form as the paper's Table 4.

        Returns
        -------
        list of dict
            One entry per layer with its name, input and output shapes written
            with ``B`` for the batch dimension, its parameter count, and its
            activation. Printing this next to Table 4 is the quickest way to
            confirm the architecture was transcribed correctly.
        """
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
