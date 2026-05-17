"""Deterministic seeding for `random`, `numpy`, and `torch` (CPU + CUDA).

Two separate Python interpreters that both call `seed_everything(seed)` and
then sample from the same RNG must produce byte-identical outputs — the §4.6
reproducibility contract.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that the project touches and (optionally) lock cuDNN /
    cuBLAS into deterministic mode.

    The `CUBLAS_WORKSPACE_CONFIG` env var must be set before the first CUDA
    matmul; we set it here so callers don't have to remember.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, AttributeError):
            # Some torch builds raise when a non-deterministic op is reachable;
            # we still want the rest of the seeding to apply.
            torch.use_deterministic_algorithms(True, warn_only=True)
