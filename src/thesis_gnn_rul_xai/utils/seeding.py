"""
Reproducible random numbers
===========================

Why this module exists
----------------------
Many steps in a machine learning pipeline use random numbers: shuffling data,
splitting a dataset into training and validation parts, initialising the
weights of a neural network. If those random numbers are different every time
the code runs, the results are different every time too, and the experiment
cannot be repeated by anyone else. That would make the work impossible to
verify.

The standard solution is a **random seed**: a fixed starting number for the
random number generator. Given the same seed, the generator produces exactly
the same sequence of numbers on every run. Reporting the seed alongside the
results is normal practice in machine learning research.

An important warning about seeds
--------------------------------
Fixing the seed makes a single run repeatable. It does **not** make the result
reliable. A result obtained from one seed can be noticeably better or worse
than the average result across seeds, purely by chance. For that reason,
experiments in this project are intended to be run with several different
seeds, reporting the mean and the standard deviation. Reporting only the single
best seed is a well-known source of over-optimistic results in the literature
and is avoided here.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["DEFAULT_SEED", "make_rng", "set_global_seed"]

#: The seed used everywhere in this project unless another value is given.
#: The specific value has no meaning; it only has to stay constant.
DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Set the seed of every global random number generator we rely on.

    This covers Python's built-in :mod:`random` module, NumPy's legacy global
    generator, and the ``PYTHONHASHSEED`` environment variable, which affects
    the iteration order of some Python data structures.

    Parameters
    ----------
    seed : int
        The seed value. The same value always produces the same sequence.

    Returns
    -------
    int
        The seed that was set, so that it can be logged or stored in a results
        file without having to repeat it in the calling code.

    Notes
    -----
    Deep learning frameworks such as PyTorch have their own generators. When a
    model is added to this project in a later stage, its seeding call belongs
    here as well, so that a single function call still covers everything.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Create an isolated NumPy random number generator.

    A local generator is preferable to the global one, because it cannot be
    disturbed by unrelated library code that also draws random numbers. Passing
    an explicit generator object to a function makes the source of randomness
    visible in the function signature, which is easier to audit.

    Parameters
    ----------
    seed : int
        Seed for this generator.

    Returns
    -------
    numpy.random.Generator
        A generator instance that is independent of the global state.
    """
    return np.random.default_rng(seed)
