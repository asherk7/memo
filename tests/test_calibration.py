"""Fusion-calibration tests (offline, synthetic).

The optimization core (`fit_fusion_scalars`) is exercised on synthetic per-modality
logits — no encoders or real data. A wiring smoke runs `run_calibrate_fusion`
end-to-end with stub encoders + a tiny aligned JSONL.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from memo.config import CalibrationConfig, ModalityDropoutConfig
from memo.encoders.base import BaseEncoder
from memo.fusion import LateFusion
from memo.labels import NUM_CLASSES
from memo.training.calibrate_fusion import fit_fusion_scalars, run_calibrate_fusion

_CALIB = CalibrationConfig(epochs=200, lr=1e-2)
_DROPOUT = ModalityDropoutConfig(rate=0.3, text_rate=0.15)


def _synthetic_logits(n: int = 512, seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Cached logits where image is strongly predictive, audio mildly, text ≈ noise."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)

    def informative(strength: float) -> torch.Tensor:
        z = torch.randn(n, NUM_CLASSES, generator=g) * 0.1
        z[torch.arange(n), labels] += strength
        return z

    logits = {
        "image": informative(4.0),
        "text": torch.randn(n, NUM_CLASSES, generator=g) * 0.5,
        "audio": informative(1.0),
    }
    return logits, labels


def test_fit_reduces_nll() -> None:
    logits, labels = _synthetic_logits()
    fusion = LateFusion()
    history = fit_fusion_scalars(
        logits, labels, fusion, calibration=_CALIB, dropout=_DROPOUT, seed=0
    )

    assert len(history) == _CALIB.epochs
    # Net decrease: late-epoch mean well below the early-epoch mean.
    assert sum(history[-10:]) / 10 < sum(history[:10]) / 10
    # Plenty of monotone-decrease steps.
    decreasing = sum(1 for prev, cur in zip(history, history[1:], strict=False) if cur < prev)
    assert decreasing >= 50


def test_weights_move_off_uniform() -> None:
    logits, labels = _synthetic_logits()
    fusion = LateFusion()  # weight_init=0 → softmax(w) starts uniform
    assert torch.allclose(fusion.weight, torch.zeros(3))

    fit_fusion_scalars(logits, labels, fusion, calibration=_CALIB, dropout=_DROPOUT, seed=0)

    assert fusion.weight.std() > 0.05  # moved off uniform
    w = F.softmax(fusion.weight.detach(), dim=0)
    # Image (idx 0, strongly predictive) outweighs the noise text channel (idx 1).
    assert float(w[0]) > float(w[1])


def test_temperatures_finite_positive() -> None:
    logits, labels = _synthetic_logits()
    fusion = LateFusion()
    fit_fusion_scalars(logits, labels, fusion, calibration=_CALIB, dropout=_DROPOUT, seed=0)

    temps = fusion.temperature
    assert torch.isfinite(temps).all()
    assert bool((temps > 0).all())


# ---------------------------------------------------------------------------
# Wiring smoke: run_calibrate_fusion end-to-end with stubs
# ---------------------------------------------------------------------------


class _StubEncoder(BaseEncoder):
    """Flatten → pad/trim to input_dim → linear → 7 logits (handles tensor or dict)."""

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
        elif t.size(1) > self.input_dim:
            t = t[:, : self.input_dim]
        return self._fc(t)


def test_calibrate_run_smoke(tmp_path: Path) -> None:
    records = [
        {
            "id": str(i),
            "image": f"{i}.jpg",
            "text": "hi",
            "audio": f"{i}.wav",
            "label": i % NUM_CLASSES,
        }
        for i in range(16)
    ]
    aligned = tmp_path / "val.jsonl"
    aligned.write_text("\n".join(json.dumps(r) for r in records))

    encoders = {m: _StubEncoder(m) for m in ("image", "text", "audio")}
    loaders: dict[str, Callable[[Any], Any]] = {
        "image": lambda _p: torch.randn(8),
        "text": lambda _s: torch.randn(8),
        "audio": lambda _p: torch.randn(8),
    }
    out = tmp_path / "fusion.pt"
    manifest_path = run_calibrate_fusion(
        aligned,
        out=out,
        device="cpu",
        runs_dir=tmp_path / "runs",
        encoders=encoders,  # type: ignore[arg-type]
        loaders=loaders,
        remap_from="ekman7",
    )

    assert out.exists()
    assert manifest_path.name == "manifest.json"
    raw = json.loads(manifest_path.read_text())
    assert "final_nll" in raw["metrics"] and "initial_nll" in raw["metrics"]

    # The checkpoint is a plain LateFusion state-dict — reloads into a fresh module.
    fresh = LateFusion()
    fresh.load_state_dict(torch.load(out, weights_only=True))
    assert sum(p.numel() for p in fresh.parameters()) == 7
