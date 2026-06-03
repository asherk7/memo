"""Audio augmentation.

Waveform-domain ops (noise, gain, time-stretch) run before the log-mel transform;
SpecAugment runs on the resulting log-mel. All are pure functions; randomness is
controllable via an injected numpy `Generator` (waveform) or torch `Generator`
(SpecAugment) for reproducibility.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

__all__ = ["add_gaussian_noise", "apply_gain", "time_stretch", "spec_augment"]


def add_gaussian_noise(
    waveform: np.ndarray,
    snr_db_range: tuple[float, float] = (10.0, 30.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add white noise at a random SNR drawn uniformly from `snr_db_range`."""
    rng = rng or np.random.default_rng()
    y = np.asarray(waveform, dtype=np.float32)
    snr_db = rng.uniform(*snr_db_range)
    signal_power = float(np.mean(y**2)) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return y + noise


def apply_gain(
    waveform: np.ndarray,
    max_db: float = 6.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scale by a random gain in ±`max_db` dB."""
    rng = rng or np.random.default_rng()
    gain_db = rng.uniform(-max_db, max_db)
    return (np.asarray(waveform, dtype=np.float32) * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def time_stretch(
    waveform: np.ndarray,
    rate_range: tuple[float, float] = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Time-stretch by a random rate (length changes; re-fixed by preprocessing)."""
    rng = rng or np.random.default_rng()
    rate = rng.uniform(*rate_range)
    return librosa.effects.time_stretch(np.asarray(waveform, dtype=np.float32), rate=rate)


def spec_augment(
    log_mel: torch.Tensor,
    n_time_masks: int = 2,
    time_mask_max: int = 30,
    n_freq_masks: int = 2,
    freq_mask_max: int = 12,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out up to `n_freq_masks` frequency bands (≤`freq_mask_max` bins) and
    `n_time_masks` time bands (≤`time_mask_max` frames) on a `(..., n_mels, T)` tensor."""
    out = log_mel.clone()
    n_mels, n_frames = out.shape[-2], out.shape[-1]

    def _rand(high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=generator).item())

    for _ in range(n_freq_masks):
        f = _rand(freq_mask_max + 1)
        if f == 0:
            continue
        f0 = _rand(max(1, n_mels - f + 1))
        out[..., f0 : f0 + f, :] = 0.0

    for _ in range(n_time_masks):
        t = _rand(time_mask_max + 1)
        if t == 0:
            continue
        t0 = _rand(max(1, n_frames - t + 1))
        out[..., :, t0 : t0 + t] = 0.0

    return out
