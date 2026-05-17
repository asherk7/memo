"""Public, immutable result type for `MultimodalEmotionPipeline.predict`.

The frozen dataclass is the user-facing contract: every field is populated
on every call (no Optionals), so downstream UI and CLI code never has to
branch on missing keys.
"""

from __future__ import annotations

from dataclasses import dataclass

from .labels import EkmanEmotion

__all__ = ["EmotionPrediction"]


@dataclass(frozen=True)
class EmotionPrediction:
    """Result of a single multimodal inference.

    Fields mirror §5.1 of ARCHITECTURE.md:
      - `label`: the argmax of `probs`.
      - `probs`: fused class distribution.
      - `per_modality_probs`: per-modality distribution after T-scaling.
      - `confidences`: c_i = 1 - H(p_i)/log(7) per modality used.
      - `gate_weights`: post-renormalization α̃_i per modality used.
      - `used_modalities`: the modalities that actually contributed
        (after silent face degradation; not necessarily what the caller passed).
      - `abstained`: True iff max(probs) < τ (config constant).
    """

    label: EkmanEmotion
    probs: dict[EkmanEmotion, float]
    per_modality_probs: dict[str, dict[EkmanEmotion, float]]
    confidences: dict[str, float]
    gate_weights: dict[str, float]
    used_modalities: tuple[str, ...]
    abstained: bool
