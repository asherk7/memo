"""Per-sample modality dropout.

Calibration must see the same missing-modality conditions it faces at inference,
so each modality is dropped independently per sample, never the whole batch at
once. A per-batch implementation would collapse the within-batch
missing-modality diversity the gate is supposed to learn from.

Drop rates are asymmetric: text drops at half the rate of the others, since text
is usually the highest-quality signal. At least one modality always survives per
sample (the fusion layer has nothing to fuse otherwise).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..config import ModalityDropoutConfig

__all__ = ["modality_keep_mask", "apply_modality_dropout"]

# Text drops at `text_rate`; every other modality drops at `rate`.
_TEXT = "text"


def modality_keep_mask(
    batch_size: int,
    modalities: Sequence[str],
    *,
    rate: float = 0.3,
    text_rate: float = 0.15,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Per-sample boolean keep mask for each modality.

    Returns ``{modality: BoolTensor(batch_size)}`` where ``True`` means the
    modality is kept for that sample. Each modality is dropped by an independent
    Bernoulli draw (text at ``text_rate``, others at ``rate``). Any sample that
    would lose every modality has one restored uniformly at random, so every
    sample keeps at least one modality.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if not modalities:
        raise ValueError("modalities must be non-empty.")

    keep: dict[str, torch.Tensor] = {}
    for m in modalities:
        drop_rate = text_rate if m == _TEXT else rate
        # Keep with probability 1 - drop_rate, drawn independently per sample.
        keep[m] = torch.rand(batch_size, generator=generator) >= drop_rate

    # Guarantee ≥1 surviving modality per sample.
    stacked = torch.stack([keep[m] for m in modalities], dim=1)  # (B, M)
    none_kept = ~stacked.any(dim=1)  # (B,)
    if bool(none_kept.any()):
        restore_idx = torch.randint(len(modalities), (batch_size,), generator=generator)
        for j, m in enumerate(modalities):
            keep[m] = keep[m] | (none_kept & (restore_idx == j))

    return keep


def modality_keep_mask_from_config(
    batch_size: int,
    modalities: Sequence[str],
    cfg: ModalityDropoutConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """`modality_keep_mask` with rates pulled from a `ModalityDropoutConfig`."""
    return modality_keep_mask(
        batch_size,
        modalities,
        rate=cfg.rate,
        text_rate=cfg.text_rate,
        generator=generator,
    )


def apply_modality_dropout(
    batch: dict[str, torch.Tensor],
    *,
    rate: float = 0.3,
    text_rate: float = 0.15,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Zero out dropped rows in a batched modality dict in-place-safe fashion.

    ``batch`` maps each present modality to a ``(B, ...)`` tensor. Dropped rows
    are zeroed (not removed) so the batch keeps a static shape for the encoders;
    the corresponding gate input is later masked by the keep mask. Returns a new
    dict; input tensors are not mutated.
    """
    if not batch:
        raise ValueError("batch must contain at least one modality.")
    batch_size = next(iter(batch.values())).size(0)
    keep = modality_keep_mask(
        batch_size, list(batch), rate=rate, text_rate=text_rate, generator=generator
    )
    out: dict[str, torch.Tensor] = {}
    for m, x in batch.items():
        mask = keep[m].view(-1, *([1] * (x.dim() - 1))).to(x.dtype)
        out[m] = x * mask
    return out
