"""Classification + calibration metrics (§6.1), implemented independently.

Everything is computed from a confusion matrix or directly from probabilities so
the math is auditable against hand-derived fixtures (tests cross-check against
scikit-learn). Inputs accept either NumPy arrays or torch tensors.

- **macro-F1**: unweighted mean per-class F1 — the imbalance-robust primary metric.
- **weighted-F1**: support-weighted mean per-class F1 (comparable to published baselines).
- **UAR**: unweighted average recall (= macro recall) — the speech-emotion standard.
- **ECE (15-bin)**: bin-level reliability gap between confidence and accuracy.
- **Brier**: strictly-proper squared error on the probability simplex.

ECE and Brier probe distinct calibration aspects, so both are reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..labels import NUM_CLASSES

__all__ = [
    "confusion_matrix",
    "per_class_prf",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "uar",
    "expected_calibration_error",
    "brier_score",
    "classification_report",
]


def _np(x: Any) -> np.ndarray:
    """Array-like (NumPy or torch) → detached NumPy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _labels_preds(probs_or_preds: Any, labels: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Normalize inputs to ``(labels, preds, probs_or_None)``.

    Accepts either hard predictions ``(N,)`` or probabilities ``(N, K)``; when
    given probabilities, ``preds = argmax``.
    """
    y = _np(labels).astype(np.int64).ravel()
    arr = _np(probs_or_preds)
    if arr.ndim == 2:
        return y, arr.argmax(axis=1).astype(np.int64), arr
    return y, arr.astype(np.int64).ravel(), None


def confusion_matrix(
    probs_or_preds: Any, labels: Any, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """``(K, K)`` integer matrix; rows are true class, columns predicted."""
    y, preds, _ = _labels_preds(probs_or_preds, labels)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, preds), 1)
    return cm


def per_class_prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall, F1 from a confusion matrix (0 where undefined)."""
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)  # column sums = predicted-as-k
    true_pos = cm.sum(axis=1).astype(np.float64)  # row sums = support of class k
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def accuracy(probs_or_preds: Any, labels: Any) -> float:
    y, preds, _ = _labels_preds(probs_or_preds, labels)
    return float((preds == y).mean()) if y.size else 0.0


def macro_f1(probs_or_preds: Any, labels: Any, num_classes: int = NUM_CLASSES) -> float:
    """Unweighted mean per-class F1, averaged over **all** ``num_classes``.

    A class absent from the slice contributes F1 = 0 (matching scikit-learn's
    ``average="macro"``). This differs deliberately from `uar`, which averages
    over *present* classes only — on a full all-present test set the two policies
    coincide, but they diverge on an incomplete-class slice.
    """
    _, _, f1 = per_class_prf(confusion_matrix(probs_or_preds, labels, num_classes))
    return float(f1.mean())


def weighted_f1(probs_or_preds: Any, labels: Any, num_classes: int = NUM_CLASSES) -> float:
    """Support-weighted mean per-class F1."""
    cm = confusion_matrix(probs_or_preds, labels, num_classes)
    _, _, f1 = per_class_prf(cm)
    support = cm.sum(axis=1).astype(np.float64)
    total = support.sum()
    return float((f1 * support).sum() / total) if total > 0 else 0.0


def uar(probs_or_preds: Any, labels: Any, num_classes: int = NUM_CLASSES) -> float:
    """Unweighted average recall (= macro recall) — averaged over classes present.

    Classes with no support are excluded from the average so an absent class can't
    drag UAR to zero on a small test slice.
    """
    cm = confusion_matrix(probs_or_preds, labels, num_classes)
    _, recall, _ = per_class_prf(cm)
    present = cm.sum(axis=1) > 0
    return float(recall[present].mean()) if present.any() else 0.0


def expected_calibration_error(probs: Any, labels: Any, n_bins: int = 15) -> float:
    """ECE: |confidence − accuracy| averaged over equal-width confidence bins.

    Confidence is the max predicted probability; samples are bucketed into
    ``n_bins`` equal-width bins over ``[0, 1]``.
    """
    p = _np(probs)
    y = _np(labels).astype(np.int64).ravel()
    if p.ndim != 2:
        raise ValueError("ECE needs probability rows (N, K), not hard predictions.")
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    n = y.size
    if n == 0:
        return 0.0
    # Bucket: floor(conf * n_bins), with conf == 1.0 folded into the last bin.
    bin_idx = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def brier_score(probs: Any, labels: Any, num_classes: int = NUM_CLASSES) -> float:
    """Multiclass Brier score: mean squared error against one-hot targets (range [0, 2])."""
    p = _np(probs)
    y = _np(labels).astype(np.int64).ravel()
    if p.ndim != 2:
        raise ValueError("Brier needs probability rows (N, K), not hard predictions.")
    if y.size == 0:
        return 0.0
    onehot = np.eye(num_classes, dtype=np.float64)[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def classification_report(
    probs: Any, labels: Any, num_classes: int = NUM_CLASSES
) -> dict[str, Any]:
    """Bundle the headline metrics for one (probs, labels) pair."""
    cm = confusion_matrix(probs, labels, num_classes)
    precision, recall, f1 = per_class_prf(cm)
    report: dict[str, Any] = {
        "accuracy": accuracy(probs, labels),
        "macro_f1": float(f1.mean()),
        "weighted_f1": weighted_f1(probs, labels, num_classes),
        "uar": uar(probs, labels, num_classes),
        "per_class_f1": f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
    }
    p = _np(probs)
    if p.ndim == 2:  # calibration metrics need probabilities, not hard preds
        report["ece"] = expected_calibration_error(probs, labels)
        report["brier"] = brier_score(probs, labels, num_classes)
    return report
