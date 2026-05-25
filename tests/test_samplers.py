"""ClassBalancedSampler tests (Cui 2019, §4.1, Phase 7)."""

from __future__ import annotations

import torch

from memo.losses import effective_number_weights
from memo.training.samplers import ClassBalancedSampler


def test_class_balanced_sampler() -> None:
    """Per-class draw frequency matches the normalized effective-number weights."""
    # Synthetic imbalance: class 0 dominates, class 2 is a small minority.
    counts = [800, 150, 50]
    labels: list[int] = []
    for cls, n in enumerate(counts):
        labels.extend([cls] * n)

    g = torch.Generator()
    g.manual_seed(0)
    sampler = ClassBalancedSampler(labels, beta=0.9999, num_samples=200_000, generator=g)

    drawn = torch.tensor(list(sampler))
    label_t = torch.tensor(labels)
    empirical = torch.tensor(
        [float((label_t[drawn] == c).float().mean()) for c in range(len(counts))]
    )

    weights = effective_number_weights(counts, beta=0.9999)
    expected = weights / weights.sum()

    assert torch.allclose(empirical, expected, atol=0.01), f"{empirical} vs {expected}"

    # The minority class is upsampled far above its raw share (50/1000 = 0.05).
    assert empirical[2] > 0.05
    assert len(sampler) == 200_000


def test_minority_upsampled_over_majority_share() -> None:
    """Effective-number weighting compresses the head class's dominance."""
    counts = [900, 100]
    labels = [0] * counts[0] + [1] * counts[1]
    g = torch.Generator()
    g.manual_seed(1)
    sampler = ClassBalancedSampler(labels, num_samples=100_000, generator=g)

    drawn = torch.tensor(list(sampler))
    label_t = torch.tensor(labels)
    minority_share = float((label_t[drawn] == 1).float().mean())

    # Raw minority share is 0.10; class-balanced sampling lifts it well above.
    assert minority_share > 0.30
