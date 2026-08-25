"""
The graph convolution layer
===========================

What a graph convolution does
-----------------------------
A graph convolution updates every node by mixing its own feature vector with
the feature vectors of its neighbours, and then applying a learned linear
transformation. One layer therefore lets a sensor "see" the sensors it is
directly connected to; stacking three layers lets it see three edges away.

The version used here is the one introduced by Kipf and Welling (2017) and
printed as Equations 8 to 10 of Wen et al. (2026)::

    H' = ReLU(A_hat @ H @ W)

where

``H``
    The current node features, shaped ``(nodes, features_in)``. For a batch of
    examples it is ``(batch, nodes, features_in)``.
``A_hat``
    The normalised adjacency matrix ``D^(-1/2) (A + I) D^(-1/2)``, computed
    once by :func:`thesis_gnn_rul_xai.graph.adjacency.normalize_adjacency`. It
    is a fixed property of the sensor graph, **not** something the network
    learns: it encodes the engineering knowledge the paper builds in.
``W``
    The learned weight matrix, shaped ``(features_in, features_out)``.

The multiplication ``A_hat @ H`` is the "mixing" step and ``@ W`` is the
"learning" step. Because ``A_hat`` is the same for every example in the batch,
the whole operation is two matrix multiplications and needs no specialised
graph library.

Why there is no bias term
-------------------------
Equations 8, 9 and 10 of the paper contain only ``W``. The two fully connected
layers that follow are written with explicit biases (their Equations 12 and
13), so the omission in the graph layers looks deliberate rather than a
shorthand. This module follows the equations as printed and offers ``bias`` as
an option that is off by default, so the effect of adding one can be measured
rather than assumed.

Why a dense matrix instead of a graph library
---------------------------------------------
Libraries such as PyTorch Geometric are built for graphs with thousands of
nodes and sparse, per-example connectivity. Here the graph has 14 nodes, is
dense enough that a sparse representation would be slower, and is identical for
every example. A plain dense multiplication is exact, has no extra dependency,
and is easier to check by hand against the paper's equations.

References
----------
Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph
convolutional networks. *International Conference on Learning Representations
(ICLR)*. https://arxiv.org/abs/1609.02907

Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
(2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
enhancement for aeroengine RUL prediction. *Aerospace Science and Technology*,
168, 110875. https://doi.org/10.1016/j.ast.2025.110875
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["GraphConvolution"]


class GraphConvolution(nn.Module):
    """One graph convolution layer: ``A_hat @ H @ W``, optionally plus a bias.

    Parameters
    ----------
    in_features : int
        Length of each node's incoming feature vector.
    out_features : int
        Length of each node's outgoing feature vector.
    bias : bool
        Whether to add a learned bias after the multiplication. Defaults to
        ``False``, because Equations 8 to 10 of Wen et al. (2026) show no bias
        term. ASSUMPTION A-14 covers the initialisation, not this choice.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The matrix ``W``, shaped ``(in_features, out_features)``.
    bias : torch.nn.Parameter or None
        The bias vector, shaped ``(out_features,)``, or ``None``.

    Notes
    -----
    The layer does not hold the adjacency matrix. It is passed to
    :meth:`forward` instead, so that one matrix can be shared by every layer of
    a model and swapped for an ablation (a fully connected graph, say) without
    rebuilding the network.

    Examples
    --------
    Two nodes joined by one edge, one feature each, and a weight matrix that
    simply copies the input. After normalisation every entry of ``A_hat`` is
    0.5, so each node ends up with the average of the two inputs:

    >>> import torch
    >>> from thesis_gnn_rul_xai.graph.adjacency import normalize_adjacency
    >>> layer = GraphConvolution(1, 1)
    >>> with torch.no_grad():
    ...     _ = layer.weight.fill_(1.0)
    >>> adjacency = torch.tensor(normalize_adjacency([[0.0, 1.0], [1.0, 0.0]]))
    >>> features = torch.tensor([[[2.0], [4.0]]])
    >>> layer(features, adjacency).squeeze().tolist()
    [3.0, 3.0]
    """

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
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the weights.

        ASSUMPTION A-14: the paper does not state how its weights are
        initialised. Glorot (Xavier) uniform initialisation is used, because it
        is what Kipf and Welling (2017) use for this exact layer, and because
        it keeps the variance of the activations roughly constant as they pass
        through the three layers, which is what makes a deep stack trainable at
        all.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # The same bound PyTorch's own Linear layer uses.
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """Apply the layer.

        Parameters
        ----------
        features : torch.Tensor
            Node features, shaped ``(batch, nodes, in_features)``.
        adjacency : torch.Tensor
            The normalised adjacency matrix, shaped ``(nodes, nodes)``.

        Returns
        -------
        torch.Tensor
            Shaped ``(batch, nodes, out_features)``.

        Raises
        ------
        ValueError
            If the shapes do not line up. The message names the mismatch,
            because a silently broadcast dimension here would connect the wrong
            sensors and still produce a plausible-looking number.
        """
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

        # A_hat @ H, then @ W. The adjacency is the same for every example, so
        # it broadcasts across the batch dimension.
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
