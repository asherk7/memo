"""Phase 5 late-fusion math tests (§3.1).

Exhaustive checks across all 7 non-empty subsets of {image, text, audio}:
renormalization invariants, the gamma=0 weighted-average reduction, abstention,
absent-modality equivalence, and the 7-scalar parameter count.
"""

from __future__ import annotations

from itertools import combinations

import torch

from memo.fusion import LateFusion
from memo.labels import NUM_CLASSES

MODALITIES = ("image", "text", "audio")


def _all_subsets() -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(MODALITIES) + 1):
        subsets.extend(combinations(MODALITIES, r))
    return subsets


def _logits(present: tuple[str, ...], batch: int = 4) -> dict[str, torch.Tensor | None]:
    return {m: (torch.randn(batch, NUM_CLASSES) if m in present else None) for m in MODALITIES}


def test_all_7_subsets_renormalize() -> None:
    torch.manual_seed(0)
    fusion = LateFusion()
    subsets = _all_subsets()
    assert len(subsets) == 7
    for present in subsets:
        out = fusion.fuse(_logits(present))
        assert out.used_modalities == present
        # Gate weights over used modalities sum to 1 per sample.
        gate_sum = torch.stack([out.gate_weights[m] for m in present], dim=-1).sum(dim=-1)
        assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-6)
        # Fused distribution sums to 1 over classes.
        assert torch.allclose(out.probs.sum(dim=-1), torch.ones(out.probs.size(0)), atol=1e-6)


def test_gamma_zero_is_weighted_average() -> None:
    torch.manual_seed(1)
    fusion = LateFusion()  # w=0 (uniform softmax), T=1 by default
    with torch.no_grad():
        fusion.gamma.fill_(0.0)

    logits = {m: torch.randn(4, NUM_CLASSES) for m in MODALITIES}
    out = fusion.fuse(logits)

    # gamma=0 → c_i^0 = 1, so gate weights collapse to softmax(w) = uniform 1/3.
    expected = torch.stack([torch.softmax(logits[m], dim=-1) for m in MODALITIES], dim=0).mean(0)
    assert torch.allclose(out.probs, expected, atol=1e-6)
    for m in MODALITIES:
        assert torch.allclose(out.gate_weights[m], torch.full((4,), 1.0 / 3.0), atol=1e-6)


def test_abstention_triggers_below_tau() -> None:
    fusion = LateFusion(abstention_threshold=0.5)
    # Uniform logits → uniform fused dist, max prob = 1/7 < 0.5 → abstain.
    uniform = {m: torch.zeros(1, NUM_CLASSES) for m in MODALITIES}
    assert fusion.fuse(uniform).abstained.all()

    # A confident single modality clears the threshold.
    confident = torch.full((1, NUM_CLASSES), -10.0)
    confident[0, 0] = 10.0
    assert not fusion.fuse({"image": confident}).abstained.any()


def test_absent_modality_no_contribution() -> None:
    torch.manual_seed(2)
    fusion = LateFusion()
    z = torch.randn(4, NUM_CLASSES)

    explicit_none = fusion.fuse({"image": z, "text": None, "audio": None})
    dropped = fusion.fuse({"image": z})

    assert explicit_none.used_modalities == dropped.used_modalities == ("image",)
    assert torch.allclose(explicit_none.probs, dropped.probs)
    assert torch.allclose(explicit_none.gate_weights["image"], dropped.gate_weights["image"])


def test_param_count_is_7() -> None:
    fusion = LateFusion()
    assert sum(p.numel() for p in fusion.parameters()) == 7
