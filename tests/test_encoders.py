"""Phase 3 encoder shape + weights tests.

Shape tests use `pretrained=False` (random init) so they need no weight
downloads. `test_image_uses_imagenet_weights` exercises the pretrained path and
skips if the weights can't be fetched (offline).
"""

from __future__ import annotations

import pytest
import torch

from memo.encoders.audio import LogMelCRNNEncoder
from memo.encoders.base import BaseEncoder, ModalityEncoder
from memo.encoders.image import MobileNetV3SmallFaceEncoder

B = 4


def _make_text_encoder(**kwargs):
    """Random-init MiniLM (config only, no weight download); skip if offline."""
    from memo.encoders.text import MiniLMTextEncoder

    try:
        return MiniLMTextEncoder(pretrained=False, **kwargs)
    except Exception as exc:  # config fetch needs network on first run
        pytest.skip(f"MiniLM config unavailable: {exc}")


def test_image_encoder_returns_B7() -> None:
    enc = MobileNetV3SmallFaceEncoder(pretrained=False)
    out = enc.predict_logits(torch.rand(B, 3, 112, 112))
    assert out.shape == (B, 7)


def test_audio_encoder_returns_B7() -> None:
    enc = LogMelCRNNEncoder()
    out = enc.predict_logits(torch.randn(B, 64, 301))
    assert out.shape == (B, 7)


def test_audio_encoder_accepts_unbatched() -> None:
    enc = LogMelCRNNEncoder()
    out = enc.predict_logits(torch.randn(64, 301))
    assert out.shape == (1, 7)


def test_text_encoder_returns_B7() -> None:
    enc = _make_text_encoder()
    batch = {
        "input_ids": torch.randint(0, 1000, (B, 16)),
        "attention_mask": torch.ones(B, 16, dtype=torch.long),
    }
    out = enc.predict_logits(batch)
    assert out.shape == (B, 7)


def test_image_uses_imagenet_weights() -> None:
    try:
        enc = MobileNetV3SmallFaceEncoder(pretrained=True)
    except Exception as exc:
        pytest.skip(f"ImageNet weights unavailable: {exc}")
    assert enc.pretrained_weights is not None


def test_no_pretrained_weights_is_none() -> None:
    enc = MobileNetV3SmallFaceEncoder(pretrained=False)
    assert enc.pretrained_weights is None


def test_encoders_satisfy_protocol() -> None:
    img = MobileNetV3SmallFaceEncoder(pretrained=False)
    aud = LogMelCRNNEncoder()
    for enc in (img, aud):
        assert isinstance(enc, BaseEncoder)
        assert isinstance(enc, ModalityEncoder)
    assert img.name == "image"
    assert aud.name == "audio"


def test_logits_are_raw_not_softmax() -> None:
    """No softmax in the encoder — raw logits differ from their softmax."""
    enc = LogMelCRNNEncoder()
    out = enc.predict_logits(torch.randn(B, 64, 301))
    assert not torch.allclose(out, out.softmax(dim=-1))
