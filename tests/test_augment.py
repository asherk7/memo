"""Phase 4 augmentation tests."""

from __future__ import annotations

import numpy as np
import torch

from memo.augment.audio import add_gaussian_noise, apply_gain, spec_augment, time_stretch
from memo.augment.image import image_train_transform, mixup
from memo.augment.text import token_dropout
from memo.seed import seed_everything

# --- image ----------------------------------------------------------------


def test_image_augment_dtype_shape() -> None:
    transform = image_train_transform()
    x = torch.rand(3, 112, 112)
    out = transform(x)
    assert out.shape == (3, 112, 112)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_mixup_linearity() -> None:
    seed_everything(0)
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 7, (8,))
    out = mixup(x, y, alpha=0.2)

    assert 0.0 <= out.lam <= 1.0
    expected = out.lam * x + (1.0 - out.lam) * x[out.permutation]
    assert torch.allclose(out.images, expected, atol=1e-6)
    assert torch.equal(out.labels_b, y[out.permutation])


def test_mixup_alpha_zero_is_identity() -> None:
    x = torch.rand(4, 3, 8, 8)
    y = torch.randint(0, 7, (4,))
    out = mixup(x, y, alpha=0.0)
    assert out.lam == 1.0
    assert torch.allclose(out.images, x)


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


# --- text -----------------------------------------------------------------


def test_token_dropout_keeps_cls_and_only_reduces_mask() -> None:
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32, dtype=torch.long)
    ids_out, mask_out = token_dropout(input_ids, attention_mask, p=0.5, generator=g)

    assert torch.equal(ids_out, input_ids)  # ids unchanged
    assert (mask_out[:, 0] == 1).all()  # [CLS] never dropped
    assert (mask_out <= attention_mask).all()  # only ever zeroes entries
    assert mask_out.sum() < attention_mask.sum()  # with p=0.5 some are dropped


def test_token_dropout_p_zero_is_identity() -> None:
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    ids_out, mask_out = token_dropout(input_ids, attention_mask, p=0.0)
    assert torch.equal(mask_out, attention_mask)
