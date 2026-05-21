"""Audio encoder: log-mel CRNN — 3-block 1D CNN → BiGRU → attention pooling → head.

Audio is sequence data, so the BiGRU stays (a pure CNN + average pool loses
utterance-scale prosody, §2.1). The CNN treats the 64 mel bins as input channels
and convolves along time; attention pooling weights expressive frames rather than
averaging uniformly.
"""

from __future__ import annotations

import torch
from torch import nn

from ..labels import NUM_CLASSES
from .base import BaseEncoder

__all__ = ["LogMelCRNNEncoder"]


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_ch),
        nn.SiLU(),
        nn.MaxPool1d(2),
    )


class LogMelCRNNEncoder(BaseEncoder):
    name = "audio"

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        n_mels: int = 64,
        gru_hidden: int = 128,
        head_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            _conv_block(n_mels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.gru = nn.GRU(128, gru_hidden, batch_first=True, bidirectional=True)
        gru_out = gru_hidden * 2
        self.attention = nn.Linear(gru_out, 1)
        self.head = nn.Sequential(
            nn.Linear(gru_out, head_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        h = self.cnn(x)  # (B, 128, T')
        h = h.transpose(1, 2)  # (B, T', 128)
        h, _ = self.gru(h)  # (B, T', 2*gru_hidden)
        # No padding mask: preprocessing fixes every clip to a 3-s window, so a
        # batch is never zero-padded to a common length (§ Phase 2 audio).
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=1)  # (B, T')
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, 2*gru_hidden)
        return self.head(pooled)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self.forward(x)
