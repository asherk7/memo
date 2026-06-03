"""Stage-1 image encoder training.

Trains `MobileNetV3SmallFaceEncoder` on FER2013.

Data directory layout expected by this command::

    <data_dir>/
        train.csv   # columns: path, label
        val.csv     # optional; if absent, 10% of train is split off

The ``path`` column is resolved relative to ``data_dir``; ``label`` is the
dataset-native integer remapped to Ekman-7 via ``--remap-from``.

Augmentation: RandAugment + flip + random erasing every epoch.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from ..augment.image import image_train_transform
from ..config import ExperimentConfig
from ..encoders.base import BaseEncoder
from ..encoders.image import MobileNetV3SmallFaceEncoder
from ..labels import EkmanEmotion, remap_fer2013
from ..preprocessing.face import FaceNotFoundError, preprocess_face
from ..seed import seed_everything
from .datasets import CsvDataset, focal_loss_from_labels, stratified_train_val_split
from .manifest import RunManifest, new_run_id
from .samplers import ClassBalancedSampler
from .trainer import Trainer

__all__ = ["run_train_image"]

_REMAPPERS: dict[str, Callable[[Any], EkmanEmotion]] = {
    "fer2013": remap_fer2013,
    "ekman7": lambda x: EkmanEmotion(int(x)),
}


def _make_image_loader(is_train: bool) -> Callable[[str], torch.Tensor]:
    """Load an image file → face-crop (112×112) + optional augmentation."""
    transform = image_train_transform() if is_train else None

    def _load(path: str) -> torch.Tensor:
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            tensor = preprocess_face(img)  # (3, 112, 112) float [0, 1]
        except FaceNotFoundError:
            logger.warning("No face in {}; using zeros", path)
            tensor = torch.zeros(3, 112, 112)
        if transform is not None:
            tensor = transform(tensor)
        return tensor

    return _load


def run_train_image(
    data_dir: Path,
    *,
    epochs: int = 15,
    out: Path,
    config: ExperimentConfig | None = None,
    device: str = "cpu",
    runs_dir: Path = Path("runs"),
    remap_from: str = "fer2013",
    val_split: float = 0.1,
    loader: Callable[[str], torch.Tensor] | None = None,
    encoder: BaseEncoder | None = None,
) -> Path:
    """Train the image encoder and write a checkpoint + manifest.

    Args:
        data_dir: directory containing ``train.csv`` (and optionally ``val.csv``).
        epochs: total training epochs.
        out: checkpoint output path.
        config: experiment config; defaults to ``ExperimentConfig()``.
        device: torch device string.
        runs_dir: root directory for run artifacts.
        remap_from: which dataset remapper to apply (``fer2013`` | ``ekman7``).
        val_split: fraction of train data to hold out when ``val.csv`` is absent.
        loader: custom image-loader for tests (receives resolved path, returns
            ``(3, 112, 112)`` tensor).  ``None`` uses the real face-preprocessing.
        encoder: encoder to train; ``None`` uses pretrained MobileNetV3-Small.

    Returns:
        Path to the written ``manifest.json``.
    """
    cfg = config or ExperimentConfig()
    cfg.train.epochs = epochs
    seed_everything(cfg.seed)

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    run_id = new_run_id("image")
    run_dir = Path(runs_dir) / run_id
    manifest = RunManifest.create(run_id, cfg, [str(data_dir)], cfg.seed)
    logger.info("image training run {} → {}", run_id, run_dir)

    # datasets
    train_loader_fn = loader if loader is not None else _make_image_loader(is_train=True)
    val_loader_fn = loader if loader is not None else _make_image_loader(is_train=False)

    train_csv = Path(data_dir) / "train.csv"
    val_csv = Path(data_dir) / "val.csv"
    full_ds = CsvDataset(train_csv, loader=train_loader_fn, remap=remap, root=data_dir)

    if val_csv.exists():
        val_ds: Subset | CsvDataset = CsvDataset(
            val_csv, loader=val_loader_fn, remap=remap, root=data_dir
        )
        train_sub: Subset | CsvDataset = full_ds
        train_labels = full_ds.labels
    else:
        train_sub, train_labels, val_ds, _ = stratified_train_val_split(
            full_ds, val_split, cfg.seed
        )

    sampler = ClassBalancedSampler(
        train_labels,
        beta=cfg.train.focal_loss.class_weight_beta,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_dl = DataLoader(train_sub, batch_size=cfg.train.batch_size, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size * 2)

    # model + loss
    enc = encoder if encoder is not None else MobileNetV3SmallFaceEncoder(pretrained=True)
    loss_fn = focal_loss_from_labels(train_labels, cfg)

    trainer = Trainer(
        enc,
        loss_fn,
        cfg.train,
        max_lr=cfg.train.scheduler.max_lr.image,
        device=device,
    )

    # train
    result = trainer.fit(train_dl, val_dl)
    logger.info("image training complete: best_val_macro_f1={:.4f}", result.best_metric or 0.0)

    out = Path(out)
    trainer.save_checkpoint(out)
    manifest.finalize(metrics={"best_val_macro_f1": result.best_metric or 0.0})
    manifest_path = manifest.write(run_dir)
    return manifest_path


def _load_image_checkpoint(path: str | Path, device: str = "cpu") -> MobileNetV3SmallFaceEncoder:
    enc = MobileNetV3SmallFaceEncoder(pretrained=False)
    enc.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return enc


# numpy array image helper (used by the CLI when accepting raw np inputs)
def _np_to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a (H, W, 3) RGB uint8 array to the face tensor."""
    return preprocess_face(arr)
