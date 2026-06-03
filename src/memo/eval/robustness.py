"""Per-sample modality-dropout robustness sweep.

Evaluates fused macro-F1 under per-sample test-time modality dropout at
increasing rates; the floor target is that fused macro-F1 stays within 5
absolute points of the all-present score under p=0.3 dropout. Operates over
cached per-modality logits (encoders frozen).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..fusion import LateFusion
from ..training.modality_dropout import modality_keep_mask
from .metrics import macro_f1

__all__ = ["modality_dropout_sweep"]


def modality_dropout_sweep(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    fusion: LateFusion,
    *,
    rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5),
    text_rate_ratio: float = 0.5,
    trials: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    """Fused macro-F1 vs per-sample modality-dropout rate.

    Args:
        logits: cached per-modality logits ``{modality: (N, 7)}``.
        labels: ``(N,)`` ground-truth labels.
        fusion: the (calibrated) `LateFusion` module.
        rates: dropout rates to sweep; ``0.0`` is the all-present baseline.
        text_rate_ratio: text drops at ``rate * ratio`` (mirrors the 0.3 / 0.15
            asymmetry used in calibration).
        trials: mask re-draws averaged per rate (dropout is stochastic).
        seed: seeds the mask generator for a reproducible sweep.

    Returns:
        ``{"per_rate_macro_f1": {rate: f1}, "baseline": f1@0, "floor_drop_at_0.3": Δ}``
        where the floor drop is ``baseline − macro_f1@0.3`` (target ≤ 0.05).
    """
    modalities = [m for m in fusion.MODALITIES if m in logits]
    if not modalities:
        raise ValueError(f"No known modalities in logits; expected some of {fusion.MODALITIES}.")
    batch = {m: logits[m] for m in modalities}
    n = labels.size(0)
    generator = torch.Generator().manual_seed(seed)

    # Always sweep rate 0.0 so the floor drop is measured against the true
    # all-present baseline, even if a caller omits it.
    sweep_rates = sorted({0.0, *(float(r) for r in rates)})

    per_rate: dict[float, float] = {}
    for rate in sweep_rates:
        if rate <= 0.0:
            per_rate[rate] = macro_f1(fusion.fuse(batch).probs, labels)  # all present
            continue
        scores = [
            macro_f1(
                fusion.fuse(
                    batch,
                    keep_mask=modality_keep_mask(
                        n,
                        modalities,
                        rate=rate,
                        text_rate=rate * text_rate_ratio,
                        generator=generator,
                    ),
                ).probs,
                labels,
            )
            for _ in range(trials)
        ]
        per_rate[rate] = float(sum(scores) / len(scores))

    baseline = per_rate[0.0]
    result: dict[str, object] = {
        "per_rate_macro_f1": {f"{r:g}": v for r, v in per_rate.items()},
        "baseline": baseline,
    }
    floor_rate = next((r for r in per_rate if abs(r - 0.3) < 1e-9), None)
    if floor_rate is not None:
        result["floor_drop_at_0.3"] = baseline - per_rate[floor_rate]
    return result
