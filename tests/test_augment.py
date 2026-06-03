"""Augmentation tests."""

from __future__ import annotations

import numpy as np
import torch

from memo.augment.audio import add_gaussian_noise, apply_gain, spec_augment, time_stretch
from memo.augment.image import image_train_transform

# --- image ----------------------------------------------------------------


def test_image_augment_dtype_shape() -> None:
    transform = image_train_transform()
    x = torch.rand(3, 112, 112)
    out = transform(x)
    assert out.shape == (3, 112, 112)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# --- audio ----------------------------------------------------------------


def test_specaugment_mask_bounds() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.ones(64, 301)
    out = spec_augment(x, generator=g)
    fully_zero_freq_bins = int((out == 0).all(dim=1).sum())
    fully_zero_time_frames = int((out == 0).all(dim=0).sum())
    assert fully_zero_freq_bins <= 2 * 12
    assert fully_zero_time_frames <= 2 * 30


def test_specaugment_preserves_shape() -> None:
    x = torch.randn(64, 301)
    assert spec_augment(x).shape == (64, 301)


def test_gaussian_noise_increases_energy() -> None:
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    noisy = add_gaussian_noise(clean, snr_db_range=(10.0, 10.0), rng=rng)
    assert noisy.shape == clean.shape
    assert np.mean(noisy**2) > np.mean(clean**2)


def test_apply_gain_scales() -> None:
    rng = np.random.default_rng(0)
    x = np.ones(1000, dtype=np.float32)
    out = apply_gain(x, max_db=6.0, rng=rng)
    # All samples scaled by the same constant factor.
    assert np.allclose(out / out[0], 1.0)


def test_time_stretch_changes_length() -> None:
    rng = np.random.default_rng(1)
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    out = time_stretch(x, rate_range=(1.1, 1.1), rng=rng)
    assert out.shape[0] != x.shape[0]
