"""memo — multimodal emotion recognition (face + text + speech → Ekman-7)."""

from __future__ import annotations

from .types import EmotionPrediction

__all__ = ["EmotionPrediction", "__version__"]

# Phase 6 will append MultimodalEmotionPipeline once `src/memo/pipeline.py` lands.

__version__ = "0.1.0"
