"""Phase 12 metric tests — hand-derived fixtures + scikit-learn cross-checks."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from memo.eval.metrics import (
    accuracy,
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    uar,
    weighted_f1,
)

# Hand fixture: 3 classes, labels [0,0,1,2], preds [0,1,1,2].
#   class0: prec 1/1, rec 1/2, f1 0.6667 ; class1: prec 1/2, rec 1/1, f1 0.6667 ; class2: 1/1/1
_LABELS = np.array([0, 0, 1, 2])
_PREDS = np.array([0, 1, 1, 2])


def test_confusion_matrix() -> None:
    cm = confusion_matrix(_PREDS, _LABELS, num_classes=3)
    expected = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(cm, expected)


def test_accuracy() -> None:
    assert accuracy(_PREDS, _LABELS) == pytest.approx(0.75)


def test_macro_f1_handcomputed() -> None:
    # (0.6667 + 0.6667 + 1.0) / 3 = 0.77778
    assert macro_f1(_PREDS, _LABELS, num_classes=3) == pytest.approx(0.77778, abs=1e-4)


def test_weighted_f1_handcomputed() -> None:
    # support [2,1,1]; (0.6667*2 + 0.6667*1 + 1*1) / 4 = 0.75
    assert weighted_f1(_PREDS, _LABELS, num_classes=3) == pytest.approx(0.75, abs=1e-4)


def test_weighted_f1_absent_class() -> None:
    # Class 2 has zero support: it contributes 0 to both numerator and the
    # total-support denominator. Pins that the denominator is support, not num_classes.
    preds = np.array([0, 1, 1, 1])
    labels = np.array([0, 0, 1, 1])
    assert weighted_f1(preds, labels, num_classes=3) == pytest.approx(0.73333, abs=1e-4)


def test_uar_handcomputed() -> None:
    # recall [0.5, 1.0, 1.0] → mean 0.8333
    assert uar(_PREDS, _LABELS, num_classes=3) == pytest.approx(0.83333, abs=1e-4)


def test_uar_averages_present_classes_only() -> None:
    # Only classes 0,1 present in a 7-class space: recall [1/2, 1] over present → 0.75,
    # NOT 0.75 * 2/7. Pins the present-only mask that is uar's reason to exist.
    preds = np.array([0, 1, 1, 1])
    labels = np.array([0, 0, 1, 1])
    assert uar(preds, labels, num_classes=7) == pytest.approx(0.75, abs=1e-9)


def test_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, size=200)
    p = rng.integers(0, 7, size=200)
    assert macro_f1(p, y) == pytest.approx(f1_score(y, p, average="macro"), abs=1e-9)
    assert weighted_f1(p, y) == pytest.approx(f1_score(y, p, average="weighted"), abs=1e-9)
    assert uar(p, y) == pytest.approx(recall_score(y, p, average="macro", zero_division=0), abs=1e-9)


def test_brier_handcomputed() -> None:
    # sample0 [0.7,0.2,0.1] label0 → 0.14 ; sample1 [0.1,0.8,0.1] label1 → 0.06 ; mean 0.10
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels, num_classes=3) == pytest.approx(0.10, abs=1e-9)


def test_brier_requires_probs() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([0, 1]), np.array([0, 1]), num_classes=3)


def test_ece_confident_half_wrong() -> None:
    # Both samples conf 0.9 (same bin); one correct, one wrong → |0.5 − 0.9| = 0.4.
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probs, labels, n_bins=15) == pytest.approx(0.4, abs=1e-9)


def test_ece_perfectly_confident_correct() -> None:
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 0])
    assert expected_calibration_error(probs, labels, n_bins=15) == pytest.approx(0.0, abs=1e-12)


def test_ece_spans_multiple_bins() -> None:
    # Two occupied bins, exercising the count/n weighting + cross-bin sum:
    #   bin 14 (conf 0.95): 1 sample, correct → gap |1.0 − 0.95|, weight 1/4
    #   bin 8  (conf 0.55): 3 samples, 2/3 correct → gap |0.667 − 0.55|, weight 3/4
    #   ECE = 0.25*0.05 + 0.75*0.11667 = 0.10
    probs = np.array([[0.95, 0.05], [0.55, 0.45], [0.55, 0.45], [0.55, 0.45]])
    labels = np.array([0, 0, 0, 1])
    assert expected_calibration_error(probs, labels, n_bins=15) == pytest.approx(0.10, abs=1e-9)
