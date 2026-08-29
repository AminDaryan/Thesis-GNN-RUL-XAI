"""Collection rules for the whole project.

The preprocessing half of this package needs only NumPy, pandas and
scikit-learn. The model half needs PyTorch, which is a large dependency and is
therefore optional (``pip install -e ".[model]"``). When it is absent, the
modules that import it are skipped during collection rather than raising an
ImportError that looks like a broken repository.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on the installed environment
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    collect_ignore_glob = [
        "src/thesis_gnn_rul_xai/models/*.py",
        "src/thesis_gnn_rul_xai/training/*.py",
        "src/thesis_gnn_rul_xai/explain/*.py",
        "tests/test_models.py",
        "tests/test_training.py",
        "tests/test_shap.py",
        "tests/test_examples_torch.py",
    ]
