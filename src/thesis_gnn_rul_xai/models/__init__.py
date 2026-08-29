"""The network itself.

Needs PyTorch: pip install -e ".[model]"

Read README.md in this folder. It explains everything these files do.
"""

from .estgnn import ESTGNN
from .layers import GraphConvolution

__all__ = ["ESTGNN", "GraphConvolution"]
