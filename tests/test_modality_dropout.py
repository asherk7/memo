"""Per-sample modality dropout tests."""

from __future__ import annotations

import torch

from memo.training.modality_dropout import apply_modality_dropout, modality_keep_mask

MODALITIES = ["image", "text", "audio"]


def _generator(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_per_sample_independence() -> None:
    """Within a batch, rows are dropped independently — not all-or-nothing."""
    keep = modality_keep_mask(8, MODALITIES, generator=_generator(0))

    # Each mask is per-sample (shape (B,)), not a single batch-wide flag.
    for m in MODALITIES:
        assert keep[m].shape == (8,)
        assert keep[m].dtype == torch.bool

    # The failure mode being guarded: a per-batch impl makes every row identical.
    stacked = torch.stack([keep[m] for m in MODALITIES], dim=1)  # (8, 3)
    distinct_rows = {tuple(row.tolist()) for row in stacked}
    assert len(distinct_rows) > 1, "all rows identical → dropout is per-batch, not per-sample"


def test_at_least_one_modality_survives() -> None:
    keep = modality_keep_mask(256, MODALITIES, generator=_generator(1))
    stacked = torch.stack([keep[m] for m in MODALITIES], dim=1)
    assert bool(stacked.any(dim=1).all()), "some sample lost every modality"


def test_asymmetric_rates() -> None:
    """Text drops at half the rate of the other modalities."""
    n = 40_000
    keep = modality_keep_mask(n, MODALITIES, rate=0.3, text_rate=0.15, generator=_generator(2))

    drop_image = float((~keep["image"]).float().mean())
    drop_audio = float((~keep["audio"]).float().mean())
    drop_text = float((~keep["text"]).float().mean())

    # Empirical rates land near their configured values (the ≥1-survivor
    # guarantee perturbs them only by ~P(all dropped)/3 ≈ 0.005).
    assert abs(drop_image - 0.30) < 0.02
    assert abs(drop_audio - 0.30) < 0.02
    assert abs(drop_text - 0.15) < 0.02

    # And text really is ~half the others.
    assert 0.4 < drop_text / drop_image < 0.6


def test_apply_zeroes_dropped_rows() -> None:
    batch = {
        "image": torch.ones(64, 3, 4, 4),
        "text": torch.ones(64, 8),
    }
    out = apply_modality_dropout(batch, generator=_generator(3))

    # The real invariant: dropout zeroes whole rows, never partially — every row
    # is either fully kept (all ones) or fully dropped (all zeros).
    for m in ("image", "text"):
        kept_rows = 0
        for i in range(64):
            row = out[m][i]
            all_one, all_zero = bool(torch.all(row == 1)), bool(torch.all(row == 0))
            assert all_one or all_zero, f"row {i} of {m} was partially zeroed"
            kept_rows += all_one
        # Some rows survive and some drop (not degenerate all-kept / all-dropped).
        assert 0 < kept_rows < 64

    # The ≥1-survivor guarantee holds across the batch.
    for i in range(64):
        assert bool(torch.any(out["image"][i] == 1)) or bool(torch.any(out["text"][i] == 1))
