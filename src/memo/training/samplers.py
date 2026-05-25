"""Class-balanced sampler (Cui et al. 2019, §4.1).

Emotion datasets are heavily skewed, so the data layer oversamples minority
classes using the same effective-number-of-samples weighting that reweights
`FocalLoss`. This *complements* the loss-side weight rather than replacing it —
the recipe Cui recommends applies both.

The probability of drawing a sample from class ``c`` is proportional to that
class's effective-number weight, so over an epoch the class histogram the model
sees matches the normalized weights regardless of the raw class counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler

from ..losses import effective_number_weights

__all__ = ["ClassBalancedSampler"]


class ClassBalancedSampler(Sampler[int]):
    """Weighted sampler whose per-class draw probability follows Cui 2019.

    Each sample of class ``c`` receives weight ``α_c / n_c`` where ``α_c`` is the
    effective-number weight and ``n_c`` the class count, so the total mass of
    class ``c`` is ``α_c`` and ``P(class c) = α_c / Σ_j α_j``. Sampling is with
    replacement (the point is to upsample minorities).
    """

    def __init__(
        self,
        labels: Sequence[int] | torch.Tensor,
        *,
        beta: float = 0.9999,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        label_t = torch.as_tensor(labels, dtype=torch.long).flatten()
        if label_t.numel() == 0:
            raise ValueError("labels must be non-empty.")

        classes, inverse, counts = torch.unique(label_t, return_inverse=True, return_counts=True)
        class_weights = effective_number_weights(counts, beta=beta)  # (C,)

        # Per-sample weight = α_c / n_c for that sample's class.
        per_sample = class_weights[inverse] / counts[inverse].to(class_weights.dtype)

        self.classes = classes
        self.weights = per_sample
        self.num_samples = num_samples if num_samples is not None else label_t.numel()
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        idx = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=self.generator
        )
        yield from idx.tolist()

    def __len__(self) -> int:
        return self.num_samples
