"""Phase 2 acceptance tests for the preprocessing layer."""

from __future__ import annotations

import librosa
import numpy as np
import pytest
import torch

from memo.preprocessing.audio import (
    N_MELS,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    fix_length,
    preprocess_audio,
    resample,
)
from memo.preprocessing.face import (
    FaceNotFoundError,
    _align_crop_resize,
    preprocess_face,
)
from memo.preprocessing.text import preprocess_text

_EXPECTED_T = 1 + TARGET_SAMPLES // 160  # librosa center=True frame count


# --- face -----------------------------------------------------------------


def test_face_not_found_raises() -> None:
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(FaceNotFoundError):
        preprocess_face(blank)


def test_align_crop_resize_shape_dtype_range() -> None:
    img = np.random.default_rng(0).integers(0, 256, (200, 200, 3)).astype(np.uint8)
    out = _align_crop_resize(
        img,
        right_eye=(70.0, 90.0),
        left_eye=(130.0, 92.0),
        bbox=(60.0, 60.0, 90.0, 90.0),
        size=112,
    )
    assert out.shape == (3, 112, 112)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


# --- audio ----------------------------------------------------------------


def test_resample_matches_librosa() -> None:
    y = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    ours = resample(y, 8000, SAMPLE_RATE)
    ref = librosa.resample(y, orig_sr=8000, target_sr=SAMPLE_RATE)
    assert np.allclose(ours, ref)
    # Same-rate is a no-op.
    assert np.array_equal(resample(y, SAMPLE_RATE, SAMPLE_RATE), y)


@pytest.mark.parametrize("seconds", [1.0, 3.0, 6.0])
def test_logmel_shape_stable(seconds: float) -> None:
    n = int(SAMPLE_RATE * seconds)
    y = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    out = preprocess_audio(y, SAMPLE_RATE)
    assert out.shape == (N_MELS, _EXPECTED_T)


def test_logmel_T_identical_across_durations() -> None:
    shapes = {
        tuple(preprocess_audio(np.zeros(int(SAMPLE_RATE * s), np.float32), SAMPLE_RATE).shape)
        for s in (1.0, 3.0, 6.0)
    }
    assert len(shapes) == 1


def test_fix_length_center_pad_and_crop() -> None:
    short = np.ones(100, dtype=np.float32)
    padded = fix_length(short, 10_000)
    assert padded.shape[0] == 10_000
    assert padded[:10].sum() == 0.0 and padded[-10:].sum() == 0.0  # zero-padded edges

    long = np.arange(20_000, dtype=np.float32)
    cropped = fix_length(long, 10_000)
    assert cropped.shape[0] == 10_000
    assert cropped[0] == 5_000  # centered crop


# --- text -----------------------------------------------------------------


def test_tokenizer_shapes() -> None:
    try:
        out = preprocess_text(
            ["short", "a noticeably longer sentence than the others", "mid length here"]
        )
    except Exception as exc:  # offline / no cached tokenizer
        pytest.skip(f"MiniLM tokenizer unavailable: {exc}")

    ids, mask = out["input_ids"], out["attention_mask"]
    assert ids.shape == mask.shape
    assert ids.shape[0] == 3
    assert ids.dim() == 2


def test_tokenizer_single_string() -> None:
    try:
        out = preprocess_text("just one string")
    except Exception as exc:
        pytest.skip(f"MiniLM tokenizer unavailable: {exc}")
    assert out["input_ids"].shape[0] == 1
