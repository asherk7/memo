"""Phase 10 joint fine-tune tests (§4.2).

Offline: three stub encoders + a real `LateFusion` exercise the multi-task loss,
per-sample modality dropout reaching the gate, the bespoke training loop, and the
per-encoder checkpoint reload through `from_config` — no datasets, no network.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from memo.encoders.base import BaseEncoder
from memo.fusion import LateFusion
from memo.labels import NUM_CLASSES
from memo.losses import FocalLoss
from memo.training.modality_dropout import modality_keep_mask
from memo.training.train_joint import joint_loss, run_train_joint

MODALITIES = ("image", "text", "audio")


class _StubEncoder(BaseEncoder):
    """Flatten any input (tensor or token dict) → linear → 7 logits."""

    def __init__(self, name: str, input_dim: int) -> None:
        super().__init__()
        self.name = name
        self.num_classes = NUM_CLASSES
        self._fc = nn.Linear(input_dim, NUM_CLASSES)

    def predict_logits(self, x: Tensor | dict[str, Tensor]) -> Tensor:
        t = x[next(iter(x))] if isinstance(x, dict) else x
        return self._fc(torch.as_tensor(t, dtype=torch.float32).flatten(1))


def _stub_encoders() -> dict[str, BaseEncoder]:
    return {
        "image": _StubEncoder("image", 3 * 112 * 112),
        "text": _StubEncoder("text", 16),
        "audio": _StubEncoder("audio", 64 * 301),
    }


def _fake_loaders() -> dict:
    return {
        "image": lambda _p: torch.randn(3, 112, 112),
        "text": lambda _s: {
            "input_ids": torch.randint(0, 100, (16,)),
            "attention_mask": torch.ones(16, dtype=torch.long),
        },
        "audio": lambda _p: torch.randn(64, 301),
    }


def _write_jsonl(path: Path, n: int) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "image": f"{i}.png",
                        "text": f"sentence {i}",
                        "audio": f"{i}.wav",
                        "label": i % NUM_CLASSES,
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------


def test_joint_loss_aux_grads() -> None:
    """Every encoder receives non-zero gradient from the multi-task loss."""
    torch.manual_seed(0)
    fusion = LateFusion()
    b = 16
    logits = {m: torch.randn(b, NUM_CLASSES, requires_grad=True) for m in MODALITIES}
    keep = modality_keep_mask(b, MODALITIES, generator=torch.Generator().manual_seed(0))
    targets = torch.randint(0, NUM_CLASSES, (b,))
    focal_none = FocalLoss(gamma=2.0, label_smoothing=0.05, reduction="none")

    fused = fusion.fuse(logits, keep_mask=keep).probs
    loss, aux = joint_loss(fused, logits, keep, targets, focal_none, lam=0.3)
    loss.backward()

    assert set(aux) == set(MODALITIES)
    for m in MODALITIES:
        g = logits[m].grad
        assert g is not None and torch.any(g != 0), f"{m} received no gradient"


def test_joint_loss_decreases() -> None:
    """A short bespoke joint loop drives the multi-task loss down."""
    torch.manual_seed(0)
    encoders = _stub_encoders()
    fusion = LateFusion()
    for p in fusion.parameters():
        p.requires_grad = False
    params = [p for e in encoders.values() for p in e.parameters()]
    opt = torch.optim.Adam(params, lr=1e-2)
    focal_none = FocalLoss(gamma=0.0, label_smoothing=0.0, reduction="none")

    b = 16
    inputs = {
        "image": torch.randn(b, 3, 112, 112),
        "text": torch.randint(0, 100, (b, 16)),
        "audio": torch.randn(b, 64, 301),
    }
    targets = torch.randint(0, NUM_CLASSES, (b,))
    gen = torch.Generator().manual_seed(0)

    losses: list[float] = []
    for _ in range(40):
        keep = modality_keep_mask(b, MODALITIES, generator=gen)
        logits = {m: encoders[m].predict_logits(inputs[m]) for m in MODALITIES}
        fused = fusion.fuse(logits, keep_mask=keep).probs
        loss, _ = joint_loss(fused, logits, keep, targets, focal_none, lam=0.3)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert (
        losses[-1] < losses[0]
    ), f"joint loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"


# ---------------------------------------------------------------------------
# Per-sample modality dropout statistics (§4.5)
# ---------------------------------------------------------------------------


def test_joint_modality_dropout_stats() -> None:
    """Empirical keep rates match the configured (asymmetric) drop rates ±5% rel."""
    gen = torch.Generator().manual_seed(0)
    kept: Counter[str] = Counter()
    total = 0
    for _ in range(400):
        keep = modality_keep_mask(64, MODALITIES, rate=0.3, text_rate=0.15, generator=gen)
        for m in MODALITIES:
            kept[m] += int(keep[m].sum())
        total += 64

    img_rate = kept["image"] / total
    txt_rate = kept["text"] / total
    aud_rate = kept["audio"] / total
    # ≥1-survivor restoration nudges keep rates slightly above (1 - drop); allow 5% rel.
    assert abs(img_rate - 0.70) / 0.70 < 0.05
    assert abs(aud_rate - 0.70) / 0.70 < 0.05
    assert abs(txt_rate - 0.85) / 0.85 < 0.05


# ---------------------------------------------------------------------------
# End-to-end run + reload
# ---------------------------------------------------------------------------


def test_joint_smoke_artifacts(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", 32)
    _write_jsonl(tmp_path / "val.jsonl", 8)
    out = tmp_path / "joint.pt"

    manifest_path = run_train_joint(
        tmp_path / "train.jsonl",
        tmp_path / "val.jsonl",
        out=out,
        epochs=4,
        remap_from="ekman7",
        loaders=_fake_loaders(),
        encoders=_stub_encoders(),
        fusion=LateFusion(),
        runs_dir=tmp_path / "runs",
    )

    assert manifest_path.name == "manifest.json"
    for m in MODALITIES:
        assert out.with_name(f"joint_{m}.pt").exists(), f"missing joint_{m}.pt"
    raw = json.loads(manifest_path.read_text())
    assert "fused_macro_f1" in raw["metrics"]
    assert raw["end_time"] is not None


def test_joint_checkpoint_reloads(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The per-encoder joint checkpoints reload through `from_config` offline."""
    from memo.pipeline import MultimodalEmotionPipeline

    _write_jsonl(tmp_path / "train.jsonl", 32)
    _write_jsonl(tmp_path / "val.jsonl", 8)
    out = tmp_path / "joint.pt"
    run_train_joint(
        tmp_path / "train.jsonl",
        tmp_path / "val.jsonl",
        out=out,
        epochs=2,
        remap_from="ekman7",
        loaders=_fake_loaders(),
        encoders=_stub_encoders(),
        fusion=LateFusion(),
        runs_dir=tmp_path / "runs",
    )

    # Point a config at the saved per-encoder checkpoints; swap the real encoder
    # classes for architecture-matching stubs so from_config stays offline.
    cfg = {
        "model": {
            "encoders": {
                "image": {"weights": None, "checkpoint": str(out.with_name("joint_image.pt"))},
                "text": {"checkpoint": str(out.with_name("joint_text.pt"))},
                "audio": {"checkpoint": str(out.with_name("joint_audio.pt"))},
            }
        }
    }
    cfg_path = tmp_path / "joint_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(
        "memo.pipeline.MobileNetV3SmallFaceEncoder",
        lambda *a, **k: _StubEncoder("image", 3 * 112 * 112),
    )
    monkeypatch.setattr("memo.pipeline.MiniLMTextEncoder", lambda *a, **k: _StubEncoder("text", 16))
    monkeypatch.setattr(
        "memo.pipeline.LogMelCRNNEncoder", lambda *a, **k: _StubEncoder("audio", 64 * 301)
    )

    pipe = MultimodalEmotionPipeline.from_config(cfg_path)
    assert isinstance(pipe, MultimodalEmotionPipeline)
    # The loaded weights match what training saved (reload is faithful).
    saved = torch.load(out.with_name("joint_audio.pt"), weights_only=True)
    loaded = pipe.audio_encoder.state_dict()["_fc.weight"]  # type: ignore[attr-defined]
    assert torch.allclose(loaded, saved["_fc.weight"])
