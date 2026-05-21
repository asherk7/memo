"""Text encoder: frozen MiniLM-L6 + 2-layer MLP head, optional LoRA r=8.

By default the MiniLM backbone is frozen and only the ~50K-param head trains.
With `lora=True`, peft attaches r=8 adapters to the last 2 transformer layers
(query/key/value/dense), adding ~110K trainable params on top of the head.

Sentence embeddings come from attention-masked mean pooling over the backbone's
*contextual* token embeddings — this is the sentence-transformers convention,
not a bag-of-embeddings mean-pool (which would discard word order and pretrained
semantics, §2.1).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..labels import NUM_CLASSES
from ..preprocessing.text import BACKBONE
from .base import BaseEncoder

__all__ = ["MiniLMTextEncoder"]

# MiniLM-L6 has 6 transformer layers; adapt the last two.
_LORA_LAYERS = [4, 5]
_LORA_TARGETS = ["query", "key", "value", "dense"]


class MiniLMTextEncoder(BaseEncoder):
    name = "text"

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        lora: bool = False,
        pretrained: bool = True,
        head_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.num_classes = num_classes
        self.lora_enabled = lora

        if pretrained:
            backbone = AutoModel.from_pretrained(BACKBONE)
        else:
            backbone = AutoModel.from_config(AutoConfig.from_pretrained(BACKBONE))

        hidden = backbone.config.hidden_size  # 384

        if lora:
            backbone = self._wrap_lora(backbone)
        else:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    @staticmethod
    def _wrap_lora(backbone: Any) -> Any:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=_LORA_TARGETS,
            layers_to_transform=_LORA_LAYERS,
            layers_pattern="layer",
        )
        return get_peft_model(backbone, config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.head(pooled)

    def predict_logits(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(x["input_ids"], x["attention_mask"])
