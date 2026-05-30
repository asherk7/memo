"""Image augmentation: RandAugment + flip + random erasing.

The per-sample transform runs on a `[0, 1]` float CHW tensor (the preprocessing
contract). RandAugment needs uint8, so the transform round-trips through uint8
internally and returns float32.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

__all__ = ["image_train_transform"]


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
