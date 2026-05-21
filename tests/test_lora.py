"""Phase 3 LoRA wiring tests for the text encoder.

Uses random-init MiniLM (config only) — param counts and adapter placement
depend on architecture, not pretrained weights.
"""

from __future__ import annotations

import re

import pytest


def _make_text_encoder(**kwargs):
    from memo.encoders.text import MiniLMTextEncoder

    try:
        return MiniLMTextEncoder(pretrained=False, **kwargs)
    except Exception as exc:
        pytest.skip(f"MiniLM config unavailable: {exc}")


def _trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _lora_layer_indices(model) -> set[int]:
    indices: set[int] = set()
    for name, _ in model.named_modules():
        if "lora_A" in name:
            match = re.search(r"\.layer\.(\d+)\.", name)
            if match:
                indices.add(int(match.group(1)))
    return indices


def test_lora_disabled_default() -> None:
    enc = _make_text_encoder()
    assert enc.lora_enabled is False
    trainable = _trainable_params(enc)
    assert 30_000 <= trainable <= 70_000, trainable


def test_lora_param_count() -> None:
    enc = _make_text_encoder(lora=True)
    assert enc.lora_enabled is True
    trainable = _trainable_params(enc)
    assert 100_000 <= trainable <= 200_000, trainable
    # Adapters attach only to the last 2 transformer layers.
    assert _lora_layer_indices(enc) == {4, 5}


def test_lora_adds_trainable_over_head_only() -> None:
    head_only = _trainable_params(_make_text_encoder())
    with_lora = _trainable_params(_make_text_encoder(lora=True))
    assert with_lora > head_only
