"""Loss functions for training (§4.1, §4.4).

`FocalLoss` bakes label smoothing into a single combined loss — focal modulation
and smoothing are applied together over a soft target distribution, not stacked
as two passes (which computes the wrong gradient at the boundary). Plain
cross-entropy is deliberately avoided; emotion datasets are heavily skewed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FocalLoss", "KDLoss", "effective_number_weights"]


def effective_number_weights(
    class_counts: Sequence[int] | torch.Tensor, beta: float = 0.9999
) -> torch.Tensor:
    """Cui et al. 2019 effective-number-of-samples class weights.

    α_c = (1 - β) / (1 - β^{n_c}), normalized to mean 1 (sum = num_classes).
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(counts)


def _smoothed_targets(
    targets: torch.Tensor, num_classes: int, eps: float, like: torch.Tensor
) -> torch.Tensor:
    """Soft target distribution: q_y = 1-ε+ε/K, q_{k≠y} = ε/K (PyTorch convention)."""
    q = torch.full_like(like, eps / num_classes)
    q.scatter_(1, targets.unsqueeze(1), 1.0 - eps + eps / num_classes)
    return q


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    r"""Focal loss with label smoothing baked in.

    Per sample: :math:`\alpha_y\,(1-p_y)^\gamma \cdot \big(-\sum_k q_k \log p_k\big)`
    — the standard true-class focal factor :math:`(1-p_y)^\gamma` applied to the
    (optionally smoothed) cross-entropy. With ``gamma=0``, ``label_smoothing=0``,
    and no class weights this reduces to cross-entropy.
    """

    class_weights: torch.Tensor | None

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.as_tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer("class_weights", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)

        q = _smoothed_targets(targets, num_classes, self.label_smoothing, log_p)
        ce = -(q * log_p).sum(dim=-1)

        p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - p_t) ** self.gamma * ce

        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]

        return _reduce(loss, self.reduction)


class KDLoss(nn.Module):
    r"""Hinton knowledge distillation (§4.4).

    :math:`\mathcal{L} = \alpha \mathcal{L}_{focal}(s, y)
    + (1-\alpha)\,\tau^2\,\mathrm{KL}(\sigma(t/\tau)\,\|\,\sigma(s/\tau))`.
    Reduces to plain `FocalLoss` when ``alpha == 1``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
        focal: FocalLoss | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.focal = focal if focal is not None else FocalLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        hard = self.focal(student_logits, targets)
        if self.alpha >= 1.0:
            return hard

        tau = self.temperature
        student_log = F.log_softmax(student_logits / tau, dim=-1)
        teacher_p = F.softmax(teacher_logits / tau, dim=-1)
        # F.kl_div(log_q, p) = KL(p || q); here q=student, p=teacher → KL(teacher || student).
        soft = F.kl_div(student_log, teacher_p, reduction="batchmean") * (tau * tau)
        return self.alpha * hard + (1.0 - self.alpha) * soft
