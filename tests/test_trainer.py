"""Shared training-loop smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from memo.config import ExperimentConfig
from memo.losses import FocalLoss
from memo.seed import seed_everything
from memo.training.trainer import Trainer, build_param_groups


def _image_slice(n: int = 32) -> TensorDataset:
    seed_everything(0)
    images = torch.randn(n, 3, 112, 112)
    labels = torch.randint(0, 7, (n,))
    return TensorDataset(images, labels)


def _train_config(epochs: int) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.train.epochs = epochs
    cfg.train.batch_size = 8
    cfg.train.freeze_backbone_epochs = 0
    return cfg


def test_trainer_loss_decreases_and_checkpoint_reloads(tmp_path: Path, dummy_image_encoder) -> None:
    seed_everything(0)
    cfg = _train_config(epochs=8)
    dataset = _image_slice(32)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    trainer = Trainer(dummy_image_encoder, FocalLoss(), cfg.train, max_lr=1e-2)
    result = trainer.fit(loader)

    # Converges: end-of-training loss is well below where it started.
    assert result.epoch_losses[-1] < result.epoch_losses[0]
    # And the within-run trend is downward, not just a lucky final epoch.
    first = sum(result.step_losses[:4]) / 4
    last = sum(result.step_losses[-4:]) / 4
    assert last < first

    # Checkpoint saves and reloads into a fresh encoder with identical outputs.
    ckpt = tmp_path / "image.pt"
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists()

    reloaded = type(dummy_image_encoder)("image")
    reloaded.load_state_dict(torch.load(ckpt, weights_only=True))

    probe = torch.randn(2, 3, 112, 112)
    trainer.model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(trainer.model.predict_logits(probe), reloaded.predict_logits(probe))


def test_trainer_early_stopping_and_macro_f1(dummy_image_encoder) -> None:
    seed_everything(1)
    cfg = _train_config(epochs=6)
    cfg.train.early_stopping_patience = 2
    train = DataLoader(_image_slice(32), batch_size=8, shuffle=True)
    val = DataLoader(_image_slice(16), batch_size=8)

    trainer = Trainer(dummy_image_encoder, FocalLoss(), cfg.train, max_lr=1e-2)
    result = trainer.fit(train, val)

    assert result.best_metric is not None
    assert result.best_epoch is not None
    assert len(result.val_macro_f1) >= 1
    # Early stopping never runs more epochs than configured.
    assert len(result.epoch_losses) <= cfg.train.epochs


def test_build_param_groups_splits_backbone_and_head(dummy_image_encoder) -> None:
    # The stub has no `backbone` → a single head-rate group.
    groups = build_param_groups(dummy_image_encoder, backbone_lr=1e-5, head_lr=1e-3)
    assert len(groups) == 1
    assert groups[0]["name"] == "head"
    assert groups[0]["lr"] == 1e-3
