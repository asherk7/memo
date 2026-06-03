"""Stage-1 text encoder training.

Trains `MiniLMTextEncoder` on GoEmotions → Ekman-7.

Data directory layout::

    <data_dir>/
        train.csv   # columns: text, label
        val.csv     # optional

The ``text`` column is the raw sentence; ``label`` is the native GoEmotions
integer (0–27) when ``--remap-from goemotions``, or 0–6 Ekman when ``ekman7``.

The backbone stays frozen throughout (only the ~50K-param head trains). The
Trainer's backbone-freeze curriculum is disabled so the encoder's own
``requires_grad`` setup (frozen MiniLM) is never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from ..config import ExperimentConfig
from ..encoders.base import BaseEncoder
from ..encoders.text import MiniLMTextEncoder
from ..labels import EkmanEmotion, remap_goemotions
from ..preprocessing.text import preprocess_text
from ..seed import seed_everything
from .datasets import CsvDataset, focal_loss_from_labels, stratified_train_val_split
from .manifest import RunManifest, new_run_id
from .samplers import ClassBalancedSampler
from .trainer import Trainer

__all__ = ["run_train_text", "text_collate_fn"]

_REMAPPERS: dict[str, Callable[[Any], EkmanEmotion]] = {
    "goemotions": remap_goemotions,
    "ekman7": lambda x: EkmanEmotion(int(x)),
}


def text_collate_fn(batch: list[tuple[str, int]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Tokenize a batch of (text, label) pairs.

    Calling the tokenizer on the full batch (rather than per-sample) ensures
    proper dynamic padding — shorter sequences in the batch are padded to the
    longest, not to the global max_length.
    """
    texts, labels = zip(*batch, strict=False)
    tokens = preprocess_text(list(texts))
    return tokens, torch.tensor(labels, dtype=torch.long)


def _default_collate(batch: list[tuple[Any, int]]) -> tuple[Any, torch.Tensor]:
    # Batch may contain pre-tokenized dicts (injected loader) or raw strings.
    first_item = batch[0][0]
    if isinstance(first_item, dict):
        # Pre-tokenized path (smoke tests / custom loaders).
        dicts, labels = zip(*batch, strict=False)
        stacked = {k: torch.stack([d[k] for d in dicts]) for k in first_item}
        return stacked, torch.tensor(labels, dtype=torch.long)
    # Raw string path: batch-tokenize for proper padding.
    return text_collate_fn(batch)


def run_train_text(
    data_dir: Path,
    *,
    epochs: int = 15,
    out: Path,
    config: ExperimentConfig | None = None,
    device: str = "cpu",
    runs_dir: Path = Path("runs"),
    remap_from: str = "goemotions",
    val_split: float = 0.1,
    loader: Callable[[str], Any] | None = None,
    collate_fn: Callable | None = None,
    encoder: BaseEncoder | None = None,
) -> Path:
    """Train the text encoder and write a checkpoint + manifest.

    Args:
        data_dir: directory containing ``train.csv`` (columns: ``text``, ``label``).
        epochs: total training epochs.
        out: checkpoint output path.
        config: experiment config; defaults to ``ExperimentConfig()``.
        device: torch device string.
        runs_dir: root directory for run artifacts.
        remap_from: ``goemotions`` | ``ekman7``.
        val_split: hold-out fraction when ``val.csv`` is absent.
        loader: custom per-sample loader (receives text string, returns string or
            pre-tokenized dict).  ``None`` uses identity (raw text → collate tokenizes).
        collate_fn: custom collate for tests (replaces the real tokenizing collate).
        encoder: encoder to train; ``None`` uses ``MiniLMTextEncoder``.

    Returns:
        Path to ``manifest.json``.
    """
    cfg = config or ExperimentConfig()
    cfg.train.epochs = epochs
    seed_everything(cfg.seed)

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    run_id = new_run_id("text")
    run_dir = Path(runs_dir) / run_id
    manifest = RunManifest.create(run_id, cfg, [str(data_dir)], cfg.seed)
    logger.info("text training run {} → {}", run_id, run_dir)

    # Default text loader is identity (raw string); batch tokenization happens
    # in the collate_fn so padding is correct for the whole batch.
    item_loader: Callable[[str], Any] = loader if loader is not None else (lambda x: x)

    train_csv = Path(data_dir) / "train.csv"
    val_csv = Path(data_dir) / "val.csv"
    full_ds = CsvDataset(
        train_csv,
        loader=item_loader,
        remap=remap,
        value_column="text",
        is_path=False,
    )

    if val_csv.exists():
        val_ds: Subset | CsvDataset = CsvDataset(
            val_csv,
            loader=item_loader,
            remap=remap,
            value_column="text",
            is_path=False,
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

    collate = collate_fn if collate_fn is not None else _default_collate

    train_dl = DataLoader(
        train_sub,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        collate_fn=collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        collate_fn=collate,
    )

    # model + loss
    enc = encoder if encoder is not None else MiniLMTextEncoder(pretrained=True)
    loss_fn = focal_loss_from_labels(train_labels, cfg)

    trainer = Trainer(
        enc,
        loss_fn,
        cfg.train,
        max_lr=cfg.train.scheduler.max_lr.text_head,
        device=device,
        # Text encoder manages its own requires_grad (frozen MiniLM backbone);
        # the Trainer must not overwrite that with its backbone-freeze curriculum.
        freeze_backbone_curriculum=False,
    )

    # train
    result = trainer.fit(train_dl, val_dl)
    logger.info("text training complete: best_val_macro_f1={:.4f}", result.best_metric or 0.0)

    out = Path(out)
    trainer.save_checkpoint(out)
    manifest.finalize(metrics={"best_val_macro_f1": result.best_metric or 0.0})
    manifest_path = manifest.write(run_dir)
    return manifest_path
