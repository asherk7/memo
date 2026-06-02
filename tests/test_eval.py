"""Phase 12 eval-orchestration tests: subset ablation, gating, robustness, wiring smoke."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from memo.eval.evaluate import evaluate_subsets, gating_comparison, run_evaluate
from memo.eval.robustness import modality_dropout_sweep
from memo.fusion import LateFusion
from memo.labels import NUM_CLASSES


def _synthetic_logits(n: int = 256, seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)

    def informative(strength: float) -> torch.Tensor:
        z = torch.randn(n, NUM_CLASSES, generator=g) * 0.1
        z[torch.arange(n), labels] += strength
        return z

    return {"image": informative(4.0), "text": informative(0.2), "audio": informative(1.0)}, labels


def test_evaluate_subsets_covers_all_seven() -> None:
    logits, labels = _synthetic_logits()
    reports = evaluate_subsets(logits, labels, LateFusion())
    assert len(reports) == 7  # 2^3 - 1 non-empty subsets
    assert "image+text+audio" in reports
    for rep in reports.values():
        assert 0.0 <= rep["macro_f1"] <= 1.0
        assert "ece" in rep and "brier" in rep


def test_gating_comparison_restores_gamma() -> None:
    logits, labels = _synthetic_logits()
    fusion = LateFusion()
    original = fusion.gamma.detach().clone()
    result = gating_comparison(logits, labels, fusion)
    assert set(result) == {"gating_on_macro_f1", "gating_off_macro_f1", "gate_gain"}
    assert 0.0 <= result["gating_on_macro_f1"] <= 1.0
    assert 0.0 <= result["gating_off_macro_f1"] <= 1.0
    # gamma must be restored to its pre-call value (in-place γ=0 then restore).
    assert torch.allclose(fusion.gamma, original)


def test_gating_comparison_gate_helps() -> None:
    """A case the confidence gate is *built* for: two uncertain modalities agree on
    the wrong class, a confident one is correct. γ>0 down-weights the uncertain
    pair (recovers the truth); γ=0 (equal weight) is out-voted."""
    n = 210
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    wrong = (labels + 1) % NUM_CLASSES
    image = torch.zeros(n, NUM_CLASSES)
    image[torch.arange(n), labels] += 3.0  # confident-correct (low entropy)
    text = torch.zeros(n, NUM_CLASSES)
    text[torch.arange(n), wrong] += 2.0  # less-confident, agree on WRONG
    audio = torch.zeros(n, NUM_CLASSES)
    audio[torch.arange(n), wrong] += 2.0

    result = gating_comparison({"image": image, "text": text, "audio": audio}, labels, LateFusion())
    assert result["gating_on_macro_f1"] > result["gating_off_macro_f1"]
    assert result["gate_gain"] > 0.0


def test_modality_dropout_sweep() -> None:
    logits, labels = _synthetic_logits()
    out = modality_dropout_sweep(logits, labels, LateFusion(), rates=(0.0, 0.3), trials=4, seed=0)
    per_rate = out["per_rate_macro_f1"]
    baseline = out["baseline"]
    assert isinstance(per_rate, dict) and isinstance(baseline, float)
    assert "0" in per_rate and "0.3" in per_rate
    assert baseline == per_rate["0"]
    assert "floor_drop_at_0.3" in out
    # Baseline (all present) is a valid macro-F1.
    assert 0.0 <= baseline <= 1.0


# --- wiring smoke ----------------------------------------------------------


class _StubEncoder(nn.Module):
    def __init__(self, name: str, input_dim: int = 8) -> None:
        super().__init__()
        self.name = name
        self.num_classes = NUM_CLASSES
        self.input_dim = input_dim
        self._fc = nn.Linear(input_dim, NUM_CLASSES)

    def predict_logits(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        t = x[next(iter(x))] if isinstance(x, dict) else x
        t = torch.as_tensor(t, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        t = t.flatten(start_dim=1)
        if t.size(1) < self.input_dim:
            t = F.pad(t, (0, self.input_dim - t.size(1)))
        return self._fc(t[:, : self.input_dim])


def test_run_evaluate_smoke(tmp_path: Path) -> None:
    records = [
        {"id": str(i), "image": "x", "text": "hi", "audio": "y", "label": i % NUM_CLASSES}
        for i in range(24)
    ]
    aligned = tmp_path / "test.jsonl"
    aligned.write_text("\n".join(json.dumps(r) for r in records))

    encoders = {m: _StubEncoder(m) for m in ("image", "text", "audio")}
    loaders: dict[str, Callable[[Any], Any]] = {
        "image": lambda _p: torch.randn(8),
        "text": lambda _s: torch.randn(8),
        "audio": lambda _p: torch.randn(8),
    }
    out_dir = tmp_path / "eval"
    report = run_evaluate(
        aligned,
        out_dir=out_dir,
        encoders=encoders,  # type: ignore[arg-type]
        fusion=LateFusion(),
        loaders=loaders,
        remap_from="ekman7",
        dropout_rates=(0.0, 0.3),
    )

    assert report["n_samples"] == 24
    assert {"headline", "subsets", "gating", "robustness"} <= set(report)
    assert len(report["subsets"]) == 7
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    # Report round-trips as valid JSON.
    json.loads((out_dir / "report.json").read_text())
