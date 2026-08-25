"""
Thesis GNN RUL XAI
==================

A reproducible pipeline for Remaining Useful Life (RUL) prediction on the
NASA C-MAPSS turbofan engine dataset. It is the software foundation for a
master's thesis on combining Graph Neural Networks with Explainable AI for
predictive maintenance.

This package is a **re-implementation study** based on:

    Wen, J.-R., Liang, C.-J., Choy, Y.-S., Fei, C.-W., Li, J., & Keshtegar, B.
    (2026). Explainable spatiotemporal graph neutral network with dynamic SHAP
    enhancement for aeroengine RUL prediction.
    *Aerospace Science and Technology*, 168, 110875.
    https://doi.org/10.1016/j.ast.2025.110875

Important scientific note
-------------------------
The original paper does not specify every implementation detail. Wherever a
value or a procedure had to be chosen by us rather than read from the paper,
the choice is marked in the source code with the tag ``ASSUMPTION`` and is
recorded in ``docs/ASSUMPTIONS.md`` together with its justification. No
undocumented guess is used anywhere in this package.

Package layout
--------------
``config``
    Typed configuration objects, loadable from a YAML file.
``data``
    Everything that turns raw C-MAPSS text files into model-ready arrays.
``graph``
    Construction of the sensor graph (the "adjacency matrix") used by the
    Graph Neural Network.
``utils``
    Cross-cutting helpers: reproducible random seeds, project paths, logging.
"""

__version__ = "0.1.0"
__author__ = "Amin Darian"
__all__ = ["__version__"]
