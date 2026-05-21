"""Modality encoders. Each returns raw `(B, 7)` logits via `predict_logits`."""

from __future__ import annotations

from .audio import LogMelCRNNEncoder
from .base import BaseEncoder, ModalityEncoder
from .image import MobileNetV3SmallFaceEncoder
from .text import MiniLMTextEncoder

__all__ = [
    "ModalityEncoder",
    "BaseEncoder",
    "MobileNetV3SmallFaceEncoder",
    "MiniLMTextEncoder",
    "LogMelCRNNEncoder",
]
