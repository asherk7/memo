"""memo — multimodal emotion recognition (face + text + speech → Ekman-7)."""

from __future__ import annotations

from .pipeline import MultimodalEmotionPipeline
from .types import EmotionPrediction

__all__ = ["EmotionPrediction", "MultimodalEmotionPipeline", "__version__"]

__version__ = "0.1.0"
