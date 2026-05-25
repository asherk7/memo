"""Stratified k-fold runner (§4.1), opt-in via ``--k-fold``.

Small audio sets (RAVDESS, 1440 clips) are noisy to evaluate on a single split,
so training can run stratified 5-fold and average the per-fold metrics. The
split is seeded so the folds are reproducible across runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

__all__ = ["stratified_folds", "run_kfold"]


def stratified_folds(
    labels: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n_splits`` ``(train_idx, val_idx)`` pairs, stratified by label."""
    y = np.asarray(labels)
    if y.size < n_splits:
        raise ValueError(f"Need ≥{n_splits} examples for {n_splits}-fold, got {y.size}.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # StratifiedKFold needs a dummy X of matching length; only y is used.
    return [(train, val) for train, val in skf.split(np.zeros(y.size), y)]


def run_kfold(
    labels: Sequence[int] | np.ndarray,
    fit_fold: Callable[[int, np.ndarray, np.ndarray], dict[str, float]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Run ``fit_fold(fold_idx, train_idx, val_idx)`` per fold and average metrics.

    ``fit_fold`` returns a metric dict per fold; the keys are averaged across
    folds (each averaged key is suffixed with ``_mean``).
    """
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        stratified_folds(labels, n_splits=n_splits, seed=seed)
    ):
        fold_metrics.append(fit_fold(fold_idx, train_idx, val_idx))

    keys = fold_metrics[0].keys() if fold_metrics else []
    return {f"{k}_mean": float(np.mean([m[k] for m in fold_metrics])) for k in keys}
