"""End-to-end inference pipeline (§1, §5.1).

`MultimodalEmotionPipeline.predict(image=, text=, audio=)` runs only the
preprocessor → encoder for the modalities the caller actually supplied, hands
the per-modality logits to `LateFusion`, and reduces the batched `FusionOutput`
into a single `EmotionPrediction`. Any non-empty subset of the three modalities
works; an all-`None` call raises.

The preprocessing functions are imported at module scope so they can be patched
in tests — the pipeline's own job is orchestration and the fusion → prediction
reduction, not preprocessing (Phase 2) or encoding (Phase 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .encoders.audio import LogMelCRNNEncoder
from .encoders.base import ModalityEncoder
from .encoders.image import MobileNetV3SmallFaceEncoder
from .encoders.text import MiniLMTextEncoder
from .fusion import FusionOutput, LateFusion
from .labels import NUM_CLASSES, EkmanEmotion
from .preprocessing.audio import SAMPLE_RATE, preprocess_audio
from .preprocessing.face import FaceNotFoundError, preprocess_face
from .preprocessing.text import preprocess_text
from .types import EmotionPrediction

__all__ = ["MultimodalEmotionPipeline"]


def _probs_to_dict(row: torch.Tensor) -> dict[EkmanEmotion, float]:
    """Map a `(7,)` probability row to `{EkmanEmotion: float}` (§5.1)."""
    return {EkmanEmotion(i): float(row[i]) for i in range(NUM_CLASSES)}


def _maybe_load(module: nn.Module, checkpoint: str | None) -> None:
    """Load a state dict into `module` if a checkpoint path is configured.

    Uses `weights_only=True` (§8) — never unpickle arbitrary objects.
    """
    if checkpoint is None:
        return
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module.load_state_dict(state)


class MultimodalEmotionPipeline:
    """Owns the three encoders and one `LateFusion`, exposes `predict`."""

    def __init__(
        self,
        image_encoder: ModalityEncoder,
        text_encoder: ModalityEncoder,
        audio_encoder: ModalityEncoder,
        fusion: LateFusion,
    ) -> None:
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.fusion = fusion
        # Inference-only: disable dropout / BN updates. `ModalityEncoder` is a
        # Protocol that doesn't require nn.Module, so guard the `.eval()` call
        # (a non-Module encoder, e.g. an ONNX-runtime one, has no train/eval).
        for enc in (image_encoder, text_encoder, audio_encoder):
            if isinstance(enc, nn.Module):
                enc.eval()
        fusion.eval()

    @classmethod
    def from_config(cls, path: str | Path) -> MultimodalEmotionPipeline:
        """Build encoders + fusion from a YAML config, loading any configured
        checkpoints (otherwise the encoders stay random/pretrained-init)."""
        cfg = ExperimentConfig.from_yaml(path)
        enc = cfg.model.encoders

        # Honor the config knobs each encoder accepts: a null `weights` disables
        # ImageNet pretraining (§2.1), `lora.enabled` flips the text adapters on,
        # and the text head dropout is forwarded.
        image_encoder = MobileNetV3SmallFaceEncoder(pretrained=bool(enc.image.weights))
        text_encoder = MiniLMTextEncoder(lora=cfg.model.lora.enabled, dropout=enc.text.head_dropout)
        audio_encoder = LogMelCRNNEncoder(n_mels=enc.audio.n_mels)

        _maybe_load(image_encoder, enc.image.checkpoint)
        _maybe_load(text_encoder, enc.text.checkpoint)
        _maybe_load(audio_encoder, enc.audio.checkpoint)

        fusion = LateFusion.from_config(cfg.model.fusion)
        _maybe_load(fusion, cfg.model.fusion.checkpoint)

        return cls(image_encoder, text_encoder, audio_encoder, fusion)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | None = None,
        text: str | None = None,
        audio: np.ndarray | None = None,
        *,
        audio_sample_rate: int = SAMPLE_RATE,
    ) -> EmotionPrediction:
        """Run inference over whichever modalities are supplied (§1).

        Raises `ValueError` if all three are `None`. If `image` is supplied but
        no face is found, the image modality is silently dropped and the
        remaining modalities are used (§1).
        """
        if image is None and text is None and audio is None:
            raise ValueError("predict() requires at least one of image, text, or audio.")

        # Only present modalities are inserted; an absent modality is simply
        # never added (LateFusion treats a missing key the same as a None value).
        logits: dict[str, torch.Tensor | None] = {}

        if image is not None:
            try:
                face = preprocess_face(image).unsqueeze(0)
            except FaceNotFoundError:
                face = None  # silent face degradation (§1)
            if face is not None:
                logits["image"] = self.image_encoder.predict_logits(face)

        if text is not None:
            logits["text"] = self.text_encoder.predict_logits(preprocess_text(text))

        if audio is not None:
            logmel = preprocess_audio(audio, audio_sample_rate).unsqueeze(0)
            logits["audio"] = self.audio_encoder.predict_logits(logmel)

        if not logits:
            # Reachable only when the sole input was an image with no detectable
            # face. Own the message here rather than leaking LateFusion's.
            raise ValueError(
                "predict() found no usable modality: the only input was an image "
                "with no detectable face. Supply text or audio alongside the image."
            )

        return self._to_prediction(self.fusion.fuse(logits))

    @staticmethod
    def _to_prediction(fused: FusionOutput) -> EmotionPrediction:
        """Reduce a batch-of-1 `FusionOutput` to a scalar `EmotionPrediction`."""
        probs_row = fused.probs[0]
        return EmotionPrediction(
            label=EkmanEmotion(int(probs_row.argmax())),
            probs=_probs_to_dict(probs_row),
            per_modality_probs={
                m: _probs_to_dict(p[0]) for m, p in fused.per_modality_probs.items()
            },
            confidences={m: float(c[0]) for m, c in fused.confidences.items()},
            gate_weights={m: float(g[0]) for m, g in fused.gate_weights.items()},
            used_modalities=fused.used_modalities,
            abstained=bool(fused.abstained[0]),
        )
