"""End-to-end pipeline tests.

Uses the conftest `DummyEncoder` stand-ins and patched preprocessing so the test
exercises pipeline orchestration + the FusionOutput → EmotionPrediction
reduction without depending on MediaPipe, the HF tokenizer, or real datasets.
The patched preprocessors return plain tensors the modality-agnostic
`DummyEncoder` can flatten.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from memo.fusion import LateFusion
from memo.labels import NUM_CLASSES, EkmanEmotion
from memo.pipeline import MultimodalEmotionPipeline, _maybe_load
from memo.preprocessing.face import FaceNotFoundError
from memo.types import EmotionPrediction

MODALITIES = ("image", "text", "audio")
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@pytest.fixture
def pipeline(  # type: ignore[no-untyped-def]
    dummy_image_encoder, dummy_text_encoder, dummy_audio_encoder
) -> MultimodalEmotionPipeline:
    return MultimodalEmotionPipeline(
        dummy_image_encoder, dummy_text_encoder, dummy_audio_encoder, LateFusion()
    )


@pytest.fixture
def patched_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the three preprocessors with deterministic shape-correct stubs."""
    monkeypatch.setattr(
        "memo.pipeline.preprocess_face", lambda image, size=112: torch.rand(3, 112, 112)
    )
    monkeypatch.setattr(
        "memo.pipeline.preprocess_text",
        lambda text, max_length=128: torch.ones(1, 8),
    )
    monkeypatch.setattr("memo.pipeline.preprocess_audio", lambda wav, sr: torch.rand(64, 301))


def _inputs_for(
    subset: tuple[str, ...],
    synthetic_image: np.ndarray,
    synthetic_text: str,
    synthetic_audio: np.ndarray,
) -> dict[str, np.ndarray | str]:
    available: dict[str, np.ndarray | str] = {
        "image": synthetic_image,
        "text": synthetic_text,
        "audio": synthetic_audio,
    }
    return {m: available[m] for m in subset}


def _assert_valid(pred: EmotionPrediction, expected_used: tuple[str, ...]) -> None:
    assert isinstance(pred, EmotionPrediction)
    assert pred.used_modalities == expected_used
    assert pred.label in EkmanEmotion
    assert len(pred.probs) == NUM_CLASSES
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-5
    assert set(pred.per_modality_probs) == set(expected_used)
    assert set(pred.gate_weights) == set(expected_used)
    assert abs(sum(pred.gate_weights.values()) - 1.0) < 1e-5
    assert isinstance(pred.abstained, bool)


def test_all_7_subsets_return_prediction(
    pipeline,
    patched_preprocessing,
    synthetic_image,
    synthetic_text,
    synthetic_audio,  # type: ignore[no-untyped-def]
) -> None:
    subsets = [s for r in range(1, 4) for s in combinations(MODALITIES, r)]
    assert len(subsets) == 7
    for subset in subsets:
        kwargs = _inputs_for(subset, synthetic_image, synthetic_text, synthetic_audio)
        _assert_valid(pipeline.predict(**kwargs), subset)


def test_all_none_raises_valueerror(pipeline) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="at least one"):
        pipeline.predict()


def test_face_failure_silent_degradation(
    pipeline,
    monkeypatch,
    synthetic_image,
    synthetic_text,  # type: ignore[no-untyped-def]
) -> None:
    # Face preprocessing fails; text is valid → text-only prediction, no raise.
    def _raise(image, size=112):  # type: ignore[no-untyped-def]
        raise FaceNotFoundError("no face")

    monkeypatch.setattr("memo.pipeline.preprocess_face", _raise)
    monkeypatch.setattr(
        "memo.pipeline.preprocess_text", lambda text, max_length=128: torch.ones(1, 8)
    )

    pred = pipeline.predict(image=synthetic_image, text=synthetic_text)
    _assert_valid(pred, ("text",))


def test_used_modalities_reflects_actual_use(
    pipeline,
    monkeypatch,
    synthetic_image,
    synthetic_text,
    synthetic_audio,  # type: ignore[no-untyped-def]
) -> None:
    # All three passed, but the face can't be found → used reflects the survivors
    # in canonical order, not the inputs the caller passed.
    def _raise(image, size=112):  # type: ignore[no-untyped-def]
        raise FaceNotFoundError("no face")

    monkeypatch.setattr("memo.pipeline.preprocess_face", _raise)
    monkeypatch.setattr(
        "memo.pipeline.preprocess_text", lambda text, max_length=128: torch.ones(1, 8)
    )
    monkeypatch.setattr("memo.pipeline.preprocess_audio", lambda wav, sr: torch.rand(64, 301))

    pred = pipeline.predict(image=synthetic_image, text=synthetic_text, audio=synthetic_audio)
    assert pred.used_modalities == ("text", "audio")


def test_image_only_face_failure_raises(pipeline, monkeypatch, synthetic_image) -> None:  # type: ignore[no-untyped-def]
    # Image is the sole input and the face can't be found → no usable modality.
    # The pipeline owns this error rather than leaking LateFusion's message.
    def _raise(image, size=112):  # type: ignore[no-untyped-def]
        raise FaceNotFoundError("no face")

    monkeypatch.setattr("memo.pipeline.preprocess_face", _raise)
    with pytest.raises(ValueError, match="no usable modality"):
        pipeline.predict(image=synthetic_image)


def test_maybe_load_roundtrip(tmp_path: Path) -> None:
    saved = nn.Linear(3, 3)
    ckpt = tmp_path / "weights.pt"
    torch.save(saved.state_dict(), ckpt)

    loaded = nn.Linear(3, 3)
    _maybe_load(loaded, str(ckpt))
    assert torch.allclose(loaded.weight, saved.weight)

    # A None checkpoint is a no-op (leaves weights untouched).
    before = loaded.weight.clone()
    _maybe_load(loaded, None)
    assert torch.allclose(loaded.weight, before)


def test_from_config_builds_pipeline(
    monkeypatch,
    dummy_image_encoder,
    dummy_text_encoder,
    dummy_audio_encoder,  # type: ignore[no-untyped-def]
) -> None:
    # Swap the real (network-fetching) encoders for the dummies so from_config's
    # wiring is exercised offline. All checkpoints in default.yaml are null.
    monkeypatch.setattr(
        "memo.pipeline.MobileNetV3SmallFaceEncoder", lambda *a, **k: dummy_image_encoder
    )
    monkeypatch.setattr("memo.pipeline.MiniLMTextEncoder", lambda *a, **k: dummy_text_encoder)
    monkeypatch.setattr("memo.pipeline.LogMelCRNNEncoder", lambda *a, **k: dummy_audio_encoder)

    pipe = MultimodalEmotionPipeline.from_config(_DEFAULT_CONFIG)
    assert isinstance(pipe, MultimodalEmotionPipeline)
    assert isinstance(pipe.fusion, LateFusion)
    assert pipe.image_encoder is dummy_image_encoder
    assert pipe.text_encoder is dummy_text_encoder
    assert pipe.audio_encoder is dummy_audio_encoder
