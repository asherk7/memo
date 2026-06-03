"""Audio preprocessing: resample to 16 kHz → fixed 3-s window → log-mel (64×T).

Stateless and deterministic — train-time augmentation lives in `augment/audio.py`.
No pre-emphasis filter (it complicates ONNX export for marginal gain).
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

__all__ = [
    "SAMPLE_RATE",
    "N_MELS",
    "WINDOW_SECONDS",
    "TARGET_SAMPLES",
    "resample",
    "fix_length",
    "log_mel_spectrogram",
    "preprocess_audio",
]

SAMPLE_RATE = 16_000
N_MELS = 64
WINDOW_SECONDS = 3.0
N_FFT = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
FMIN = 0.0
FMAX = SAMPLE_RATE / 2
TARGET_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 48_000
_LOG_EPS = 1e-6


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return librosa.to_mono(y)
    raise ValueError(f"Expected 1-D or 2-D audio, got shape {y.shape}.")


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample mono audio to `target_sr` using librosa."""
    y = _to_mono(waveform)
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def fix_length(waveform: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """Center-pad (with zeros) or center-crop to exactly `target_len` samples."""
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"Expected mono 1-D audio, got shape {y.shape}.")
    n = y.shape[0]
    if n == target_len:
        return y
    if n > target_len:
        start = (n - target_len) // 2
        return y[start : start + target_len]
    pad_total = target_len - n
    left = pad_total // 2
    return np.pad(y, (left, pad_total - left), mode="constant")


def log_mel_spectrogram(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a `(N_MELS, T)` log-mel spectrogram via librosa."""
    y = np.asarray(waveform, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=True,  # lock frame count; T = 1 + n_samples // hop
    )
    return np.log(mel + _LOG_EPS).astype(np.float32)


def preprocess_audio(waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
    """Full path: resample → 3-s window → log-mel. Returns a `(N_MELS, T)` tensor
    with `T` independent of input duration."""
    y = resample(waveform, sample_rate, SAMPLE_RATE)
    y = fix_length(y, TARGET_SAMPLES)
    log_mel = log_mel_spectrogram(y, SAMPLE_RATE)
    return torch.from_numpy(log_mel)
