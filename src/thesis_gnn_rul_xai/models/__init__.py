"""
Model definitions
=================

The network architecture of Wen et al. (2026), Table 4: three graph
convolution layers over the fixed sensor graph, then two fully connected
layers ending in a sigmoid.

The per-sensor LSTM stage that precedes the graph layers in their full
architecture is **not** implemented here; its size, depth and output
projection are never stated in the paper. Results from this package are
therefore comparable with their "ESTGNN w/o LSTM" ablation only. See
assumptions A-07 and A-11 in ``docs/ASSUMPTIONS.md``.

This subpackage needs PyTorch, which the preprocessing half of the project
does not. Install it with the ``model`` extra::

    pip install -e ".[model]"
"""

from .estgnn import ESTGNN
from .layers import GraphConvolution

__all__ = ["ESTGNN", "GraphConvolution"]
