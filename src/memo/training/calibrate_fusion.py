"""Fusion calibration — `memo calibrate`.

Fits the 7 `LateFusion` scalars (3 temperatures, 3 weights, 1 sharpness) under
per-sample modality dropout against an aligned validation set, so the gate
performs across all 2^3 - 1 = 7 modality subsets rather than only the
all-present case.

Encoders are frozen for the entire run, so their per-modality logits are
constants w.r.t. the fusion scalars. `precompute_logits` runs each encoder over
the aligned set once, then `fit_fusion_scalars` optimizes the 7 scalars over the
cached logits, so ~200 epochs converge in seconds on CPU.

The optimization core (`fit_fusion_scalars`) takes cached logits + labels and is
exercisable offline on synthetic data; `run_calibrate_fusion` wires the frozen
encoders + aligned JSONL around it for the real run.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from ..config import CalibrationConfig, ExperimentConfig, ModalityDropoutConfig
from ..encoders.base import BaseEncoder
from ..fusion import LateFusion
from ..labels import EkmanEmotion, remap_ravdess
from ..seed import seed_everything
from .datasets import JsonlDataset
from .manifest import RunManifest, new_run_id
from .modality_dropout import modality_keep_mask_from_config

__all__ = [
    "fit_fusion_scalars",
    "precompute_logits",
    "run_calibrate_fusion",
    "default_aligned_loaders",
]

# Floor for log of the fused probability before NLL — guards log(0) without
# perturbing a real probability (orders of magnitude below any class mass).
_NLL_EPS = 1e-12

Loader = Callable[[Any], Any]


def _ekman7_remap(label: int | str) -> EkmanEmotion:
    """Accept an Ekman-7 integer (0-6) or the emotion name (e.g. ``"happiness"``)."""
    if isinstance(label, str):
        return EkmanEmotion[label.upper()]
    return EkmanEmotion(int(label))


# Aligned-set label remappers (the RAVDESS builder writes Ekman-7 ints by default).
_REMAPPERS: dict[str, Callable[[Any], EkmanEmotion]] = {
    "ekman7": _ekman7_remap,
    "ravdess": remap_ravdess,
}


# ---------------------------------------------------------------------------
# Optimization core (offline-testable on synthetic logits)
# ---------------------------------------------------------------------------


def fit_fusion_scalars(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    fusion: LateFusion,
    *,
    calibration: CalibrationConfig,
    dropout: ModalityDropoutConfig,
    seed: int = 42,
) -> list[float]:
    """Fit the 7 fusion scalars by NLL under per-sample modality dropout.

    Args:
        logits: cached per-modality logits, ``{modality: (N, 7)}`` — treated as
            constants (encoders are frozen).
        labels: ``(N,)`` ground-truth Ekman labels.
        fusion: the `LateFusion` module whose 7 scalars are optimized in place.
        calibration: epochs + lr (defaults: 200 epochs, lr 1e-2).
        dropout: per-sample modality-dropout rates (0.3, text 0.15).
        seed: seeds the dropout-mask generator for a reproducible fit.

    Returns:
        Per-epoch training NLL — a monotone-ish decreasing history.
    """
    modalities = [m for m in fusion.MODALITIES if m in logits]
    if not modalities:
        raise ValueError(f"No known modalities in logits; expected some of {fusion.MODALITIES}.")

    n = labels.size(0)
    for m in modalities:
        if logits[m].size(0) != n:
            raise ValueError(f"logits[{m!r}] has {logits[m].size(0)} rows but labels has {n}.")

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=calibration.lr)
    generator = torch.Generator().manual_seed(seed)
    batch = {m: logits[m] for m in modalities}

    history: list[float] = []
    fusion.train()
    for _ in range(calibration.epochs):
        keep = modality_keep_mask_from_config(n, modalities, dropout, generator=generator)
        optimizer.zero_grad()
        out = fusion.fuse(batch, keep_mask=keep)
        loss = F.nll_loss(out.probs.clamp_min(_NLL_EPS).log(), labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


# ---------------------------------------------------------------------------
# Encoder logit precompute (run frozen encoders over the aligned set, once)
# ---------------------------------------------------------------------------


def _batched(x: Any, device: torch.device) -> Any:
    """Add a batch dim and move to device, mirroring `pipeline.predict`.

    Text inputs are already batched dicts (`(1, L)`); image/audio are bare
    tensors that need a leading batch dim.
    """
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.unsqueeze(0).to(device)


def precompute_logits(
    dataset: JsonlDataset,
    encoders: Mapping[str, BaseEncoder],
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run each frozen encoder over the aligned set once → cached logits + labels.

    Every record must carry all of ``encoders``' modalities (calibration fuses
    over the full set, then masks per sample). Returns
    ``({modality: (N, 7)}, labels (N,))`` on CPU.
    """
    dev = torch.device(device)
    for enc in encoders.values():
        if isinstance(enc, nn.Module):
            enc.to(dev).eval()

    per_mod: dict[str, list[torch.Tensor]] = {m: [] for m in encoders}
    labels: list[int] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs, label = dataset[i]
            labels.append(int(label))
            for m, enc in encoders.items():
                x = inputs.get(m)
                if x is None:
                    raise ValueError(
                        f"Calibration needs every modality present; record {i} is missing {m!r}."
                    )
                logit = enc.predict_logits(_batched(x, dev))  # (1, 7)
                per_mod[m].append(logit.squeeze(0).cpu())

    cached = {m: torch.stack(rows) for m, rows in per_mod.items()}
    return cached, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Real-run helpers (only hit when encoders / loaders are not injected)
# ---------------------------------------------------------------------------


def _build_encoders(cfg: ExperimentConfig) -> dict[str, BaseEncoder]:
    """Build the three encoders with their stage-1 checkpoints (real-run path)."""
    from ..encoders.audio import LogMelCRNNEncoder
    from ..encoders.image import MobileNetV3SmallFaceEncoder
    from ..encoders.text import MiniLMTextEncoder
    from ..pipeline import _maybe_load

    enc_cfg = cfg.model.encoders
    image = MobileNetV3SmallFaceEncoder(pretrained=bool(enc_cfg.image.weights))
    text = MiniLMTextEncoder(dropout=enc_cfg.text.head_dropout)
    audio = LogMelCRNNEncoder(n_mels=enc_cfg.audio.n_mels)
    _maybe_load(image, enc_cfg.image.checkpoint)
    _maybe_load(text, enc_cfg.text.checkpoint)
    _maybe_load(audio, enc_cfg.audio.checkpoint)
    return {"image": image, "text": text, "audio": audio}


def default_aligned_loaders() -> dict[str, Loader]:
    """Real preprocessing loaders for an aligned JSONL (shared by calibrate + evaluate)."""
    from ..preprocessing.audio import SAMPLE_RATE, preprocess_audio
    from ..preprocessing.face import preprocess_face
    from ..preprocessing.text import preprocess_text

    def load_image(path: str) -> torch.Tensor:
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return preprocess_face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def load_audio(path: str) -> torch.Tensor:
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return preprocess_audio(waveform, sr if sr else SAMPLE_RATE)

    return {"image": load_image, "text": preprocess_text, "audio": load_audio}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_calibrate_fusion(
    aligned_val: Path,
    *,
    out: Path,
    config: ExperimentConfig | None = None,
    device: str = "cpu",
    runs_dir: Path = Path("runs"),
    modality_dropout: float | None = None,
    remap_from: str = "ekman7",
    encoders: dict[str, BaseEncoder] | None = None,
    loaders: dict[str, Loader] | None = None,
) -> Path:
    """Calibrate the 7 fusion scalars and write `fusion.pt` + a run manifest.

    Args:
        aligned_val: aligned multimodal JSONL (every record has all 3 modalities).
        out: output path for the calibrated `LateFusion` state-dict.
        config: experiment config; defaults to `ExperimentConfig()`.
        device: torch device string.
        runs_dir: root directory for run artifacts.
        modality_dropout: overrides `config.train.modality_dropout.rate` when given
            (text keeps its own asymmetric rate).
        remap_from: aligned-label remapper (``ekman7`` | ``ravdess``).
        encoders: inject frozen encoders (tests); ``None`` builds the real three.
        loaders: inject per-modality loaders (tests); ``None`` uses real preprocessing.

    Returns:
        Path to the written ``manifest.json``.
    """
    cfg = config or ExperimentConfig()
    if modality_dropout is not None:
        # Copy before mutating so a caller-supplied config is never altered.
        cfg = copy.deepcopy(cfg)
        cfg.train.modality_dropout.rate = modality_dropout
    seed_everything(cfg.seed)

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    run_id = new_run_id("calibrate")
    run_dir = Path(runs_dir) / run_id
    manifest = RunManifest.create(run_id, cfg, [str(aligned_val)], cfg.seed)
    logger.info("fusion calibration run {} → {}", run_id, run_dir)

    encoders = encoders or _build_encoders(cfg)
    loaders = loaders or default_aligned_loaders()
    dataset = JsonlDataset(aligned_val, loaders=loaders, remap=remap)
    cached, labels = precompute_logits(dataset, encoders, device=device)
    logger.info("precomputed logits for {} aligned samples", labels.size(0))

    fusion = LateFusion.from_config(cfg.model.fusion)
    history = fit_fusion_scalars(
        cached,
        labels,
        fusion,
        calibration=cfg.train.calibration,
        dropout=cfg.train.modality_dropout,
        seed=cfg.seed,
    )
    logger.info("calibration NLL {:.4f} → {:.4f}", history[0], history[-1])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fusion.state_dict(), out)
    manifest.finalize(metrics={"initial_nll": history[0], "final_nll": history[-1]})
    return manifest.write(run_dir)
