"""Image augmentation: RandAugment + flip + random erasing, plus Mixup.

The per-sample transform runs on a `[0, 1]` float CHW tensor (the preprocessing
contract). RandAugment needs uint8, so the transform round-trips through uint8
internally and returns float32. Mixup is a separate batch-level op (image-only,
enabled only for the last 50% of epochs — that schedule is the trainer's call).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Beta
from torchvision.transforms import v2

__all__ = ["image_train_transform", "mixup", "MixupBatch"]


def image_train_transform(
    randaugment_n: int = 2, randaugment_m: int = 9, erasing_p: float = 0.25
) -> v2.Compose:
    """Per-sample augmentation for face crops. float[0,1] CHW → float[0,1] CHW."""
    return v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomErasing(p=erasing_p),
        ]
    )


@dataclass
class MixupBatch:
    images: torch.Tensor
    labels_a: torch.Tensor
    labels_b: torch.Tensor
    lam: float
    permutation: torch.Tensor


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2) -> MixupBatch:
    """Convex-combine each sample with a permuted partner.

    Returns the mixed images plus both label sets and λ; the loss is
    ``λ·L(pred, labels_a) + (1-λ)·L(pred, labels_b)``. Uses the global torch RNG
    (seed via `seed_everything`).
    """
    lam = float(Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return MixupBatch(mixed, labels, labels[perm], lam, perm)
