"""Image encoder: ImageNet-pretrained MobileNetV3-Small + projection + 7-way head.

ImageNet pretraining is the single largest accuracy lever; the weights enum is
named explicitly so a refactor can't silently drop it to `None`. ImageNet
mean/std normalization happens here (the encoder owns its model-specific
normalization; preprocessing emits a plain [0, 1] tensor).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from ..labels import NUM_CLASSES
from .base import BaseEncoder

__all__ = ["MobileNetV3SmallFaceEncoder"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MobileNetV3SmallFaceEncoder(BaseEncoder):
    name = "image"

    # Annotated so type checkers see the registered buffers as Tensors.
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        pretrained: bool = True,
        proj_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None

        backbone = mobilenet_v3_small(weights=self.pretrained_weights)
        in_features = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

        self.register_buffer("image_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.image_mean) / self.image_std
        return self.head(self.backbone(x))

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
