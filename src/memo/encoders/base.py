"""Encoder interface.

`ModalityEncoder` is the structural Protocol the fusion/pipeline layers type
against. `BaseEncoder` is the concrete `nn.Module` base the three real encoders
inherit — it pins `predict_logits` as the inference entry point and centralizes
the (Phase 13) ONNX export hook so the three encoders don't each reimplement it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from ..labels import NUM_CLASSES

__all__ = ["ModalityEncoder", "BaseEncoder"]


@runtime_checkable
class ModalityEncoder(Protocol):
    name: str
    num_classes: int

    def predict_logits(self, x: Any) -> torch.Tensor:
        """Return raw `(B, 7)` logits — no softmax (that lives in LateFusion)."""
        ...

    def to_onnx(self, path: Path, quantize: bool = False) -> None: ...


class BaseEncoder(nn.Module):
    name: str = "base"
    num_classes: int = NUM_CLASSES

    def predict_logits(self, x: Any) -> torch.Tensor:
        raise NotImplementedError

    def to_onnx(self, path: Path, quantize: bool = False) -> None:
        # FP32 + dynamic-INT8 export with parity checks is Phase 13 (export.py).
        raise NotImplementedError("ONNX export is implemented in Phase 13 (export.py).")
