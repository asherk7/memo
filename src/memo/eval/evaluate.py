"""`memo evaluate` orchestration (§6) — headline metrics, 7-subset ablation,
gating-on-vs-off, and the robustness sweep, written to a markdown + JSON report.

Reuses `precompute_logits` (frozen encoders → cached logits) so the whole report
is computed from one encoder pass over the aligned test set. The ablation fuses
each of the 2^3 − 1 = 7 non-empty modality subsets; the gating comparison shows
the value of the learned sharpness γ (γ learned vs γ = 0).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from ..fusion import LateFusion
from ..training.calibrate_fusion import default_aligned_loaders, precompute_logits
from ..training.datasets import JsonlDataset
from .metrics import classification_report, macro_f1
from .robustness import modality_dropout_sweep

__all__ = ["evaluate_subsets", "gating_comparison", "run_evaluate"]

Loader = Callable[[Any], Any]
_DEFAULT_RATES = (0.0, 0.1, 0.2, 0.3, 0.5)


def evaluate_subsets(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    fusion: LateFusion,
) -> dict[str, dict[str, Any]]:
    """Classification report for every non-empty subset of the present modalities."""
    present = [m for m in fusion.MODALITIES if m in logits]
    reports: dict[str, dict[str, Any]] = {}
    for r in range(1, len(present) + 1):
        for subset in combinations(present, r):
            out = fusion.fuse({m: logits[m] for m in subset})
            reports["+".join(subset)] = classification_report(out.probs, labels)
    return reports


def gating_comparison(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    fusion: LateFusion,
) -> dict[str, float]:
    """All-present fused macro-F1 with the learned γ vs γ = 0 (gating off).

    γ = 0 collapses the confidence gate to a plain learned weighted average, so
    the gap quantifies what the confidence sharpness actually buys.
    """
    present = {m: logits[m] for m in fusion.MODALITIES if m in logits}
    # In-place ops on the leaf γ parameter require no_grad (also correct for eval).
    with torch.no_grad():
        on = macro_f1(fusion.fuse(present).probs, labels)
        saved = fusion.gamma.detach().clone()
        try:
            fusion.gamma.zero_()
            off = macro_f1(fusion.fuse(present).probs, labels)
        finally:
            fusion.gamma.copy_(saved)
    return {"gating_on_macro_f1": on, "gating_off_macro_f1": off, "gate_gain": on - off}


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Evaluation report", "", f"Samples: **{report['n_samples']}**", ""]
    head = report["headline"]
    lines += [
        "## Headline (all modalities)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| macro-F1 | {head['macro_f1']:.4f} |",
        f"| weighted-F1 | {head['weighted_f1']:.4f} |",
        f"| UAR | {head['uar']:.4f} |",
        f"| accuracy | {head['accuracy']:.4f} |",
        f"| ECE (15-bin) | {head.get('ece', float('nan')):.4f} |",
        f"| Brier | {head.get('brier', float('nan')):.4f} |",
        "",
        "## Modality ablation (macro-F1 per subset)",
        "",
        "| subset | macro-F1 |",
        "|---|---|",
    ]
    for subset, rep in report["subsets"].items():
        lines.append(f"| {subset} | {rep['macro_f1']:.4f} |")
    g = report["gating"]
    lines += [
        "",
        "## Confidence gate (γ learned vs γ=0)",
        "",
        f"- gating on: **{g['gating_on_macro_f1']:.4f}**",
        f"- gating off: {g['gating_off_macro_f1']:.4f}",
        f"- gate gain: {g['gate_gain']:+.4f}",
        "",
        "## Robustness (fused macro-F1 vs modality-dropout rate)",
        "",
        "| rate | macro-F1 |",
        "|---|---|",
    ]
    for rate, f1 in report["robustness"]["per_rate_macro_f1"].items():
        lines.append(f"| {rate} | {f1:.4f} |")
    if "floor_drop_at_0.3" in report["robustness"]:
        lines += [
            "",
            f"Floor drop at p=0.3: **{report['robustness']['floor_drop_at_0.3']:+.4f}** (target ≤ 0.05).",
        ]
    return "\n".join(lines) + "\n"


@torch.no_grad()
def run_evaluate(
    aligned_test: Path,
    *,
    out_dir: Path,
    config_path: Path | None = None,
    encoders: dict[str, Any] | None = None,
    fusion: LateFusion | None = None,
    loaders: dict[str, Loader] | None = None,
    device: str = "cpu",
    remap_from: str = "ekman7",
    dropout_rates: tuple[float, ...] = _DEFAULT_RATES,
) -> dict[str, Any]:
    """Evaluate a calibrated pipeline on an aligned test set → ``report.{json,md}``.

    Either supply ``config_path`` (loads encoders + calibrated fusion via
    `MultimodalEmotionPipeline.from_config`) or inject ``encoders`` + ``fusion``
    (tests). Returns the report dict and writes it under ``out_dir``.
    """
    from ..training.calibrate_fusion import _REMAPPERS

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    if encoders is None or fusion is None:
        if config_path is None:
            raise ValueError("run_evaluate needs either config_path or (encoders + fusion).")
        from ..pipeline import MultimodalEmotionPipeline

        pipe = MultimodalEmotionPipeline.from_config(config_path)
        encoders = {
            "image": pipe.image_encoder,
            "text": pipe.text_encoder,
            "audio": pipe.audio_encoder,
        }
        fusion = pipe.fusion

    loaders = loaders or default_aligned_loaders()
    dataset = JsonlDataset(aligned_test, loaders=loaders, remap=remap)
    logits, labels = precompute_logits(dataset, encoders, device=device)
    logger.info("evaluating {} aligned test samples", labels.size(0))

    present = {m: logits[m] for m in fusion.MODALITIES if m in logits}
    report: dict[str, Any] = {
        "n_samples": int(labels.size(0)),
        "headline": classification_report(fusion.fuse(present).probs, labels),
        "subsets": evaluate_subsets(logits, labels, fusion),
        "gating": gating_comparison(logits, labels, fusion),
        "robustness": modality_dropout_sweep(logits, labels, fusion, rates=dropout_rates),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(_render_markdown(report))
    return report
