"""Text augmentation: token dropout (p=0.05).

Drops attended tokens by zeroing their attention mask (tokenizer-agnostic — no
need for a [MASK] id). The [CLS] anchor at position 0 is never dropped. EDA-style
synonym/swap/delete augmentation is deliberately omitted (§4.1: ~30 MB WordNet
dependency for ~1 pt gain).
"""

from __future__ import annotations

import torch

__all__ = ["token_dropout"]


def token_dropout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    p: float = 0.05,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop attended tokens (sets their attention mask to 0)."""
    if p <= 0.0:
        return input_ids, attention_mask

    rand = torch.rand(attention_mask.shape, generator=generator, device=attention_mask.device)
    drop = (rand < p) & attention_mask.bool()
    drop[:, 0] = False  # keep [CLS]

    new_mask = attention_mask.clone()
    new_mask[drop] = 0
    return input_ids, new_mask
