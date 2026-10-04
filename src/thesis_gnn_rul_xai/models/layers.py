"""One layer of the network: mix each sensor with its neighbours, then reweigh.

Read README.md in this folder. It explains everything this file does.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["GraphConvolution"]


class GraphConvolution(nn.Module):
    """One graph convolution layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError(
                "in_features and out_features must both be at least 1, got "
                f"{in_features} and {out_features}."
            )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        # ASSUMPTION A-12: no bias term in these layers. The paper's Equations
        # 8 to 10 print none, while its dense layers do print one, so we follow
        # what is printed. Pass bias=True to measure what it would change.
        # Written up in docs/assumptions/A-12.md.
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Set the starting values of the weights, before any training.

        ASSUMPTION A-14: pick the starting weights at random from the range the
        Glorot rule gives, which is set by how many numbers enter and leave the
        layer. The paper never says what it starts from. PyTorch spells the
        same rule xavier_uniform_, after Xavier Glorot, who proposed it — the
        two names mean one method. Explained in README.md in this folder;
        written up in docs/assumptions/A-14.md.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # The same bound PyTorch's own Linear layer uses.
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """Run one batch through the layer."""
        if features.dim() != 3:
            raise ValueError(
                "features must be shaped (batch, nodes, in_features), got "
                f"{tuple(features.shape)}."
            )
        if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f"adjacency must be a square matrix, got {tuple(adjacency.shape)}."
            )
        if features.shape[1] != adjacency.shape[0]:
            raise ValueError(
                f"features have {features.shape[1]} node(s) but the adjacency "
                f"matrix describes {adjacency.shape[0]}."
            )
        if features.shape[2] != self.in_features:
            raise ValueError(
                f"this layer expects {self.in_features} feature(s) per node, "
                f"got {features.shape[2]}."
            )

        # A_hat @ H, then @ W, in the order the paper's Equation 8 prints it.
        # One graph serves the whole batch: the same 14x14 matrix is applied to
        # every example in it, so it needs no copy per example.
        mixed = adjacency @ features
        output = mixed @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:  # pragma: no cover - display only
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
