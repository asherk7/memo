"""Phase 4 loss math tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from memo.losses import FocalLoss, KDLoss, LabelSmoothingCE, effective_number_weights


def test_focal_gamma0_equals_ce() -> None:
    logits = torch.randn(8, 7)
    targets = torch.randint(0, 7, (8,))
    fl = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, targets)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_label_smoothing_closed_form() -> None:
    # 1 sample, 3 classes, logits [2,1,0], target 0, eps 0.1.
    # Hand-derived loss = 0.50760 (see ROADMAP Phase 4 notes).
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    targets = torch.tensor([0])
    loss = LabelSmoothingCE(label_smoothing=0.1)(logits, targets)
    assert torch.allclose(loss, torch.tensor(0.50760), atol=1e-4)
    # Cross-check against PyTorch's own label smoothing.
    torch_ref = F.cross_entropy(logits, targets, label_smoothing=0.1)
    assert torch.allclose(loss, torch_ref, atol=1e-6)


def test_focal_smoothing_equals_labelsmoothingce_at_gamma0() -> None:
    logits = torch.randn(16, 7)
    targets = torch.randint(0, 7, (16,))
    fl = FocalLoss(gamma=0.0, label_smoothing=0.05)(logits, targets)
    ls = LabelSmoothingCE(label_smoothing=0.05)(logits, targets)
    assert torch.allclose(fl, ls, atol=1e-6)


def test_kd_alpha1_reduces_to_focal() -> None:
    focal = FocalLoss()
    kd = KDLoss(alpha=1.0, focal=focal)
    student = torch.randn(8, 7)
    teacher = torch.randn(8, 7)
    targets = torch.randint(0, 7, (8,))
    assert torch.allclose(kd(student, targets, teacher), focal(student, targets))


def test_kd_soft_term_zero_when_student_equals_teacher() -> None:
    focal = FocalLoss()
    kd = KDLoss(alpha=0.5, temperature=4.0, focal=focal)
    logits = torch.randn(8, 7)
    targets = torch.randint(0, 7, (8,))
    # student == teacher → KL term is 0 → loss = alpha * focal.
    total = kd(logits, targets, logits.clone())
    expected = 0.5 * focal(logits, targets)
    assert torch.allclose(total, expected, atol=1e-6)


def test_effective_number_weights() -> None:
    counts = [1000, 100, 10]
    w = effective_number_weights(counts, beta=0.9999)
    # Rarer classes get larger weights; normalized to mean 1 (sum == num classes).
    assert w[2] > w[1] > w[0]
    assert torch.allclose(w.sum(), torch.tensor(3.0), atol=1e-5)


def test_focal_downweights_easy_examples() -> None:
    # A confident-correct example should incur less focal loss than plain CE.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    focal = FocalLoss(gamma=2.0, label_smoothing=0.0, reduction="none")(logits, targets)
    ce = F.cross_entropy(logits, targets, reduction="none")
    assert (focal < ce).all()
