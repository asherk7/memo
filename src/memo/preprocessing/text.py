"""Text preprocessing: the MiniLM tokenizer, cached as a module-level singleton.

The tokenizer is built once on first use; per-call cost is just the tokenizer
`__call__`. Returns `input_ids` / `attention_mask` ready for the text encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

__all__ = ["preprocess_text", "BACKBONE"]

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_LENGTH = 128
_tokenizer: Any = None


def _get_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    return _tokenizer


def preprocess_text(
    text: str | Sequence[str], max_length: int = _MAX_LENGTH
) -> dict[str, torch.Tensor]:
    """Tokenize a string or batch of strings into padded `(B, L)` tensors."""
    batch = [text] if isinstance(text, str) else list(text)
    if not batch:
        raise ValueError("preprocess_text received an empty batch.")

    enc = _get_tokenizer()(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
