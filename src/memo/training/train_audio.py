"""Stage-1 audio encoder training (§4.1, §8).

Trains `LogMelCRNNEncoder` on RAVDESS + CREMA-D.

Data directory layout::

    <data_dir>/
        train.csv   # columns: path, label
        val.csv     # optional

The ``path`` column resolves relative to ``data_dir``; ``label`` is the
dataset-native code (RAVDESS int 1-8, CREMA-D string "ANG" etc.) when
the corresponding ``--remap-from`` flag is used, or 0-6 Ekman with
``--remap-from ekman7``.

Use ``--k-fold`` for small sets (RAVDESS, ≈1 440 clips): runs stratified
5-fold CV, saves each fold checkpoint, copies the best-fold checkpoint to
``--out``, and records averaged metrics in the manifest.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from ..augment.audio import add_gaussian_noise, apply_gain, spec_augment, time_stretch
from ..config import ExperimentConfig
from ..encoders.audio import LogMelCRNNEncoder
from ..encoders.base import BaseEncoder
from ..labels import EkmanEmotion, remap_cremad, remap_ravdess
from ..preprocessing.audio import preprocess_audio
from ..seed import seed_everything
from .datasets import CsvDataset, focal_loss_from_labels, stratified_train_val_split
from .kfold import run_kfold
from .manifest import RunManifest, new_run_id
from .samplers import ClassBalancedSampler
from .trainer import Trainer

__all__ = ["run_train_audio"]

_REMAPPERS: dict[str, Callable] = {
    "ravdess": remap_ravdess,
    "cremad": remap_cremad,
    "ekman7": lambda x: EkmanEmotion(int(x)),
}


def _make_audio_loader(is_train: bool) -> Callable[[str], torch.Tensor]:
    """Load a WAV file → optional waveform augmentation → log-mel spectrogram."""
    rng = np.random.default_rng()

    def _load(path: str) -> torch.Tensor:
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # to mono
        if is_train:
            waveform = add_gaussian_noise(waveform, rng=rng)
            waveform = apply_gain(waveform, rng=rng)
            waveform = time_stretch(waveform, rng=rng)
        tensor = preprocess_audio(waveform, sr)  # (n_mels, T)
        if is_train:
            tensor = spec_augment(tensor)
        return tensor

    return _load


def _build_trainer(
    enc: BaseEncoder,
    labels: list[int],
    cfg: ExperimentConfig,
    device: str,
) -> Trainer:
    return Trainer(
        enc,
        focal_loss_from_labels(labels, cfg),
        cfg.train,
        max_lr=cfg.train.scheduler.max_lr.audio,
        device=device,
    )


def run_train_audio(
    data_dir: Path,
    *,
    epochs: int = 15,
    out: Path,
    k_fold: bool = False,
    config: ExperimentConfig | None = None,
    device: str = "cpu",
    runs_dir: Path = Path("runs"),
    remap_from: str = "ravdess",
    val_split: float = 0.1,
    loader: Callable[[str], torch.Tensor] | None = None,
    encoder: BaseEncoder | None = None,
    k_fold_splits: int = 5,
) -> Path:
    """Train the audio encoder and write a checkpoint + manifest.

    Args:
        data_dir: directory with ``train.csv`` (columns: ``path``, ``label``).
        epochs: total epochs per run (or per fold when ``k_fold=True``).
        out: checkpoint output path.
        k_fold: if True, run stratified ``k_fold_splits``-fold CV.
        config: experiment config; defaults to ``ExperimentConfig()``.
        device: torch device string.
        runs_dir: root directory for run artifacts.
        remap_from: ``ravdess`` | ``cremad`` | ``ekman7``.
        val_split: hold-out fraction when ``val.csv`` is absent (single-run only).
        loader: custom loader (receives resolved path, returns ``(n_mels, T)``
            tensor). ``None`` uses the real WAV-load + augment + preprocess path.
        encoder: encoder instance; ``None`` builds a fresh ``LogMelCRNNEncoder``.
        k_fold_splits: number of CV folds (injectable for tests).

    Returns:
        Path to ``manifest.json``.
    """
    cfg = config or ExperimentConfig()
    cfg.train.epochs = epochs
    seed_everything(cfg.seed)

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    run_id = new_run_id("audio")
    run_dir = Path(runs_dir) / run_id
    manifest = RunManifest.create(run_id, cfg, [str(data_dir)], cfg.seed)
    logger.info("audio training run {} → {}", run_id, run_dir)

    # ---- dataset --------------------------------------------------------
    train_loader_fn = loader if loader is not None else _make_audio_loader(is_train=True)
    val_loader_fn = loader if loader is not None else _make_audio_loader(is_train=False)

    train_csv = Path(data_dir) / "train.csv"
    val_csv = Path(data_dir) / "val.csv"
    full_ds = CsvDataset(train_csv, loader=train_loader_fn, remap=remap, root=data_dir)
    # Build the val-loader dataset once; Subsets index into it per fold.
    val_ds_full = CsvDataset(train_csv, loader=val_loader_fn, remap=remap, root=data_dir)

    out = Path(out)

    if k_fold:
        # ----- k-fold path -----------------------------------------------
        best: dict[str, float | int] = {"metric": -1.0, "fold": 0}

        def fit_fold(fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, float]:
            fold_labels = [full_ds.labels[i] for i in train_idx]
            sampler = ClassBalancedSampler(
                fold_labels,
                beta=cfg.train.focal_loss.class_weight_beta,
                generator=torch.Generator().manual_seed(cfg.seed + fold_idx),
            )
            train_dl = DataLoader(
                Subset(full_ds, train_idx.tolist()),
                batch_size=cfg.train.batch_size,
                sampler=sampler,
            )
            val_dl = DataLoader(
                Subset(val_ds_full, val_idx.tolist()),
                batch_size=cfg.train.batch_size * 2,
            )

            fold_enc = LogMelCRNNEncoder()
            trainer = _build_trainer(fold_enc, fold_labels, cfg, device)
            result = trainer.fit(train_dl, val_dl)
            metric = result.best_metric or 0.0

            fold_ckpt = run_dir / f"fold_{fold_idx}.pt"
            run_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(fold_ckpt)
            logger.info("fold {} val_macro_f1={:.4f}", fold_idx, metric)

            if metric > float(best["metric"]):
                best["metric"] = metric
                best["fold"] = fold_idx
            return {"val_macro_f1": metric}

        agg = run_kfold(full_ds.labels, fit_fold, n_splits=k_fold_splits, seed=cfg.seed)

        best_fold_ckpt = run_dir / f"fold_{int(best['fold'])}.pt"
        shutil.copy(best_fold_ckpt, out)
        manifest.finalize(
            metrics={
                **agg,
                "best_val_macro_f1": float(best["metric"]),
                "best_fold": int(best["fold"]),
            }
        )

    else:
        # ----- single-run path -------------------------------------------
        if val_csv.exists():
            val_ds: Subset | CsvDataset = CsvDataset(
                val_csv, loader=val_loader_fn, remap=remap, root=data_dir
            )
            train_sub: Subset | CsvDataset = full_ds
            train_labels = full_ds.labels
        else:
            train_sub, train_labels, val_ds, _ = stratified_train_val_split(full_ds, val_split, cfg.seed)

        sampler = ClassBalancedSampler(
            train_labels,
            beta=cfg.train.focal_loss.class_weight_beta,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        train_dl = DataLoader(train_sub, batch_size=cfg.train.batch_size, sampler=sampler)
        val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size * 2)

        enc = encoder if encoder is not None else LogMelCRNNEncoder()
        trainer = _build_trainer(enc, train_labels, cfg, device)
        result = trainer.fit(train_dl, val_dl)
        logger.info("audio training complete: best_val_macro_f1={:.4f}", result.best_metric or 0.0)

        trainer.save_checkpoint(out)
        manifest.finalize(metrics={"best_val_macro_f1": result.best_metric or 0.0})

    manifest_path = manifest.write(run_dir)
    return manifest_path
