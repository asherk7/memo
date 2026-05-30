"""Confidence-gated late fusion (§3.1) — the conceptual centerpiece.

`LateFusion` carries exactly 7 trainable scalars: a temperature and a weight
per modality (3 + 3) plus one sharpness ``gamma``. The abstention threshold is a
config constant, not a learned parameter.

For each present modality i the gate computes a temperature-scaled distribution
``p_i``, a normalized-inverse-entropy confidence ``c_i``, then mixes the
distributions with weights ``softmax(w)_i · c_i^gamma`` renormalized over the
*present* modalities. Absent modalities (``None`` in the input dict) are dropped
explicitly via a presence mask — never inferred from all-zero logits — so the
fused output depends only on the modalities that actually contributed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import FusionConfig
from .labels import NUM_CLASSES

__all__ = ["LateFusion", "FusionOutput"]

# Renorm-denominator floor: guards the degenerate all-confidence-zero case
# (every present modality maximally uncertain with gamma>0) from a 0/0 NaN.
# Orders of magnitude below any real gate weight, so it never perturbs the
# renormalization invariant the tests assert.
_EPS = 1e-12


@dataclass(frozen=True)
class FusionOutput:
    """Batched fusion result; the pipeline (Phase 6) reduces it per sample.

    Tensor shapes are batch-first: ``probs`` is ``(B, 7)``, the per-modality
    dicts are keyed by the modalities that contributed and hold ``(B, 7)``
    distributions / ``(B,)`` scalars, and ``abstained`` is ``(B,)`` bool.
    """

    probs: torch.Tensor
    per_modality_probs: dict[str, torch.Tensor]
    confidences: dict[str, torch.Tensor]
    gate_weights: dict[str, torch.Tensor]
    used_modalities: tuple[str, ...]
    abstained: torch.Tensor


class LateFusion(nn.Module):
    """Confidence-gated convex combination of per-modality distributions.

    Temperatures are stored in log-space so they remain strictly positive under
    gradient calibration (Phase 11) — ``exp`` is monotone and unconstrained, so
    AdamW can never drive a temperature to zero or negative and break the
    softmax. The 7 trainable scalars are ``_log_temperature`` (3), ``weight``
    (3), and ``gamma`` (1).
    """

    MODALITIES: tuple[str, ...] = ("image", "text", "audio")

    def __init__(
        self,
        abstention_threshold: float = 0.40,
        gamma_init: float = 1.0,
        temperature_init: float = 1.0,
        weight_init: float = 0.0,
    ) -> None:
        super().__init__()
        n = len(self.MODALITIES)
        self._log_temperature = nn.Parameter(
            torch.full((n,), math.log(temperature_init), dtype=torch.float32)
        )
        self.weight = nn.Parameter(torch.full((n,), weight_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.abstention_threshold = abstention_threshold
        self._index = {name: i for i, name in enumerate(self.MODALITIES)}

    @property
    def temperature(self) -> torch.Tensor:
        """The 3 effective temperatures ``T_i = exp(log T_i)`` — always positive."""
        return self._log_temperature.exp()

    @classmethod
    def from_config(cls, cfg: FusionConfig) -> LateFusion:
        return cls(
            abstention_threshold=cfg.abstention_threshold,
            gamma_init=cfg.gamma_init,
            temperature_init=cfg.temperature_init,
            weight_init=cfg.weight_init,
        )

    def fuse(
        self,
        per_modality_logits: Mapping[str, torch.Tensor | None],
        *,
        keep_mask: Mapping[str, torch.Tensor] | None = None,
    ) -> FusionOutput:
        """Fuse per-modality logits into a gated class distribution.

        ``keep_mask`` (optional, keyword-only) carries **per-sample** presence:
        ``{modality: BoolTensor(B)}`` where ``False`` removes that modality from
        a sample's gate (``m_i = 0`` for that row). It is the mechanism Phase 10
        joint fine-tuning and Phase 11 calibration use to apply per-sample
        modality dropout at the gate — presence stays explicit rather than being
        inferred from zeroed logits. ``keep_mask=None`` is bit-identical to the
        all-present batch-level path, so the Phase 5 tests are unaffected.
        """
        unknown = set(per_modality_logits) - set(self.MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities {sorted(unknown)}; expected {self.MODALITIES}.")

        # Presence is explicit: a key mapping to None is identical to an absent
        # key (both yield m_i = 0). Iterate in canonical order for determinism;
        # the walrus binds the narrowed (non-None) tensor for the loop below.
        present = [(m, z) for m in self.MODALITIES if (z := per_modality_logits.get(m)) is not None]
        if not present:
            raise ValueError("fuse() requires at least one non-None modality.")

        temperature = self.temperature  # (3,) — exp computed once, not per modality
        softmax_w = F.softmax(self.weight, dim=0)
        gamma = self.gamma

        probs: dict[str, torch.Tensor] = {}
        confidences: dict[str, torch.Tensor] = {}
        alphas: list[torch.Tensor] = []
        for m, z in present:
            idx = self._index[m]
            log_p = F.log_softmax(z / temperature[idx], dim=-1)
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1)
            c = (1.0 - entropy / math.log(NUM_CLASSES)).clamp(0.0, 1.0)

            probs[m] = p
            confidences[m] = c
            alpha = softmax_w[idx] * c.pow(gamma)
            if keep_mask is not None and m in keep_mask:
                # Per-sample presence m_i ∈ {0, 1}: a dropped row contributes
                # nothing to its own gate (renormalized away below).
                alpha = alpha * keep_mask[m].to(alpha.dtype)
            alphas.append(alpha)

        # (B, |S|) gate logits → renormalize over present modalities.
        alpha = torch.stack(alphas, dim=-1)
        alpha_tilde = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(_EPS)

        # (B, |S|, 7) weighted by (B, |S|, 1) → fused (B, 7).
        used = tuple(m for m, _ in present)
        stacked_p = torch.stack([probs[m] for m in used], dim=1)
        p_final = (alpha_tilde.unsqueeze(-1) * stacked_p).sum(dim=1)

        abstained = p_final.max(dim=-1).values < self.abstention_threshold
        gate_weights = {m: alpha_tilde[:, j] for j, m in enumerate(used)}

        return FusionOutput(
            probs=p_final,
            per_modality_probs=probs,
            confidences=confidences,
            gate_weights=gate_weights,
            used_modalities=used,
            abstained=abstained,
        )
