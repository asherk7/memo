"""Shared pytest fixtures: synthetic image/text/audio inputs + dummy encoders.

Lets smoke tests pull realistic-shape inputs without real datasets, and lets the
fusion tests verify the math against random-init `(B, 7)` logits.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from memo.labels import NUM_CLASSES

# --- Synthetic input fixtures --------------------------------------------


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Random 224×224 RGB uint8 image (typical pre-MediaPipe input)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_text() -> str:
    return "I am feeling happy today."


@pytest.fixture
def synthetic_audio() -> np.ndarray:
    """3 seconds of 16 kHz mono float32 noise."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(16000 * 3).astype(np.float32)


@pytest.fixture
def synthetic_logmel() -> torch.Tensor:
    """(B=1, mels=64, T=64) log-mel tensor."""
    return torch.randn(1, 64, 64)


# --- Dummy encoder used by fusion tests -----------------------------------


class DummyEncoder(nn.Module):
    """A stand-in `ModalityEncoder`-compatible module.

    Returns `(B, 7)` logits via a single linear projection over flattened input.
    """

    def __init__(self, name: str, input_dim: int = 64) -> None:
        super().__init__()
        self.name = name
        self.num_classes = NUM_CLASSES
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, NUM_CLASSES)

    def predict_logits(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        t = t.flatten(start_dim=1)
        if t.size(1) < self.input_dim:
            t = nn.functional.pad(t, (0, self.input_dim - t.size(1)))
        elif t.size(1) > self.input_dim:
            t = t[:, : self.input_dim]
        return self.proj(t)


@pytest.fixture
def dummy_image_encoder() -> DummyEncoder:
    return DummyEncoder("image")


@pytest.fixture
def dummy_text_encoder() -> DummyEncoder:
    return DummyEncoder("text")


@pytest.fixture
def dummy_audio_encoder() -> DummyEncoder:
    return DummyEncoder("audio")
