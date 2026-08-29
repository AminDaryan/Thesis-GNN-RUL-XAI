"""Fixes random numbers so a run can be repeated exactly.

Read README.md in this folder. It explains everything this file does, and why
a number from one seed is not a result.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["DEFAULT_SEED", "make_rng", "set_global_seed"]

# Used unless another seed is given. The value has no meaning; it stays constant.
DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python's random module, NumPy and PYTHONHASHSEED; return the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Create a NumPy random number generator of its own. Prefer this one."""
    return np.random.default_rng(seed)
