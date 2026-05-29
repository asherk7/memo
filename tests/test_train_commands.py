"""Smoke tests for Phase 8 per-modality training commands (§11.3).

All tests are offline: real encoders/loaders are replaced with tiny stubs so
MediaPipe, the MiniLM tokenizer, and librosa are never invoked. The focus is
on the full pipeline — CSV loading → DataLoader → Trainer → checkpoint
→ manifest + model_card — not on the preprocessing stack (tested in Phase 2-3).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from memo.encoders.base import BaseEncoder
from memo.labels import NUM_CLASSES

# ---------------------------------------------------------------------------
# Shared stub encoder (accepts anything flat-able to a tensor)
# ---------------------------------------------------------------------------


class _StubEncoder(BaseEncoder):
    """Minimal encoder: flattens input → linear → 7 logits."""

    def __init__(self, name: str, input_dim: int = 64) -> None:
        super().__init__()
        self.name = name
        self.num_classes = NUM_CLASSES
        self._fc = nn.Linear(input_dim, NUM_CLASSES)

    def predict_logits(self, x: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(x, dict):
            t = x[next(iter(x))].float()
        else:
            t = torch.as_tensor(x, dtype=torch.float32)
        return self._fc(t.flatten(1))


def _make_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _assert_run_produced_artifacts(runs_dir: Path, ckpt: Path) -> None:
    """Shared assertion: checkpoint exists + manifest + model_card written."""
    assert ckpt.exists(), f"Checkpoint missing: {ckpt}"
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {run_dirs}"
    run = run_dirs[0]
    assert (run / "manifest.json").exists()
    assert (run / "model_card.md").exists()
    # Manifest round-trip
    raw = json.loads((run / "manifest.json").read_text())
    assert raw["seed"] == 42
    assert raw["end_time"] is not None


# ---------------------------------------------------------------------------
# Image smoke test
# ---------------------------------------------------------------------------


def test_train_image_smoke(tmp_path: Path) -> None:
    from memo.training.train_image import run_train_image

    def _fake_image_loader(_path: str) -> Tensor:
        return torch.randn(3, 112, 112)

    n = 32
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(
        data_dir / "train.csv",
        [{"path": f"{i}.png", "label": i % NUM_CLASSES} for i in range(n)],
        ["path", "label"],
    )

    ckpt = tmp_path / "image.pt"
    enc = _StubEncoder("image", input_dim=3 * 112 * 112)
    manifest_path = run_train_image(
        data_dir,
        epochs=4,
        out=ckpt,
        remap_from="ekman7",
        loader=_fake_image_loader,
        encoder=enc,
        runs_dir=tmp_path / "runs",
    )

    _assert_run_produced_artifacts(tmp_path / "runs", ckpt)

    # Checkpoint reloads into a fresh encoder with identical outputs.
    enc2 = _StubEncoder("image", input_dim=3 * 112 * 112)
    enc2.load_state_dict(torch.load(ckpt, weights_only=True))
    probe = torch.randn(2, 3, 112, 112)
    enc.eval()
    enc2.eval()
    with torch.no_grad():
        assert torch.allclose(enc.predict_logits(probe), enc2.predict_logits(probe))

    # Manifest path returned correctly.
    assert manifest_path.name == "manifest.json"


# ---------------------------------------------------------------------------
# Text smoke test
# ---------------------------------------------------------------------------


def _fake_text_loader(_text: str) -> dict[str, Tensor]:
    """Return pre-tokenized tensors of fixed length — no real tokenizer needed."""
    return {
        "input_ids": torch.randint(0, 100, (16,)),
        "attention_mask": torch.ones(16, dtype=torch.long),
    }


def _stack_dict_collate(batch: list) -> tuple[dict[str, Tensor], Tensor]:
    dicts, labels = zip(*batch, strict=False)
    stacked = {k: torch.stack([d[k] for d in dicts]) for k in dicts[0]}
    return stacked, torch.tensor(labels, dtype=torch.long)


def test_train_text_smoke(tmp_path: Path) -> None:
    from memo.training.train_text import run_train_text

    n = 32
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(
        data_dir / "train.csv",
        [{"text": f"sentence {i}", "label": i % NUM_CLASSES} for i in range(n)],
        ["text", "label"],
    )

    ckpt = tmp_path / "text.pt"
    enc = _StubEncoder("text", input_dim=16)
    manifest_path = run_train_text(
        data_dir,
        epochs=4,
        out=ckpt,
        remap_from="ekman7",
        loader=_fake_text_loader,
        collate_fn=_stack_dict_collate,
        encoder=enc,
        runs_dir=tmp_path / "runs",
    )

    _assert_run_produced_artifacts(tmp_path / "runs", ckpt)
    assert manifest_path.name == "manifest.json"


def test_train_text_lora_flag_accepted(tmp_path: Path) -> None:
    """--lora=True with an injected stub encoder should not raise."""
    from memo.training.train_text import run_train_text

    n = 16
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(
        data_dir / "train.csv",
        [{"text": f"s {i}", "label": i % NUM_CLASSES} for i in range(n)],
        ["text", "label"],
    )
    enc = _StubEncoder("text", input_dim=16)
    run_train_text(
        data_dir,
        epochs=2,
        out=tmp_path / "text_lora.pt",
        lora=True,
        remap_from="ekman7",
        loader=_fake_text_loader,
        collate_fn=_stack_dict_collate,
        encoder=enc,  # stub ignores lora flag, exercises the code path
        runs_dir=tmp_path / "runs",
    )
    assert (tmp_path / "text_lora.pt").exists()


# ---------------------------------------------------------------------------
# Audio smoke test
# ---------------------------------------------------------------------------


def test_train_audio_smoke_single(tmp_path: Path) -> None:
    from memo.training.train_audio import run_train_audio

    def _fake_audio_loader(_path: str) -> Tensor:
        return torch.randn(64, 301)

    n = 32
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(
        data_dir / "train.csv",
        [{"path": f"{i}.wav", "label": i % NUM_CLASSES} for i in range(n)],
        ["path", "label"],
    )

    ckpt = tmp_path / "audio.pt"
    enc = _StubEncoder("audio", input_dim=64 * 301)
    manifest_path = run_train_audio(
        data_dir,
        epochs=4,
        out=ckpt,
        remap_from="ekman7",
        loader=_fake_audio_loader,
        encoder=enc,
        runs_dir=tmp_path / "runs",
    )

    _assert_run_produced_artifacts(tmp_path / "runs", ckpt)
    assert manifest_path.name == "manifest.json"


def test_train_audio_smoke_kfold(tmp_path: Path) -> None:
    from memo.training.train_audio import run_train_audio

    def _fake_audio_loader_kf(_path: str) -> Tensor:
        return torch.randn(64, 301)

    n = 40

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Ensure all 7 classes present for stratification.
    labels = [i % NUM_CLASSES for i in range(n)]
    _make_csv(
        data_dir / "train.csv",
        [{"path": f"{i}.wav", "label": labels[i]} for i in range(n)],
        ["path", "label"],
    )

    ckpt = tmp_path / "audio_kf.pt"
    run_train_audio(
        data_dir,
        epochs=2,
        out=ckpt,
        k_fold=True,
        remap_from="ekman7",
        loader=_fake_audio_loader_kf,
        # No injected encoder — k-fold builds a fresh one per fold.
        runs_dir=tmp_path / "runs",
        k_fold_splits=2,
    )

    assert ckpt.exists(), "Best-fold checkpoint not copied to --out"
    run_dirs = list((tmp_path / "runs").iterdir())
    run = run_dirs[0]
    assert (run / "manifest.json").exists()
    raw = json.loads((run / "manifest.json").read_text())
    assert "val_macro_f1_mean" in raw["metrics"], raw["metrics"]
    assert "best_fold" in raw["metrics"]


# ---------------------------------------------------------------------------
# Mixup hook unit test
# ---------------------------------------------------------------------------


def test_mixup_hook_loss_math() -> None:
    """The hook computes λ·L(a) + (1-λ)·L(b) in the Mixup half of training."""
    from memo.losses import FocalLoss
    from memo.training.train_image import _make_mixup_hook

    torch.manual_seed(0)
    loss_fn = FocalLoss(gamma=0.0, label_smoothing=0.0)
    total_epochs = 4
    hook = _make_mixup_hook(loss_fn, total_epochs, alpha=0.0)
    # alpha=0 → the guard `if alpha > 0 else 1.0` returns lam=1 (no mixing).
    # Beta(0, 0) itself would raise; the guard in augment/image.py prevents that.
    enc = _StubEncoder("image", input_dim=3 * 4 * 4)
    inputs = torch.randn(8, 3, 4, 4)
    targets = torch.randint(0, NUM_CLASSES, (8,))

    # Early epoch: standard loss path.
    loss_early = hook(enc, inputs, targets, 0)
    assert loss_early.ndim == 0 and loss_early.item() > 0

    # Late epoch (epoch >= total_epochs // 2 = 2): Mixup path.
    loss_late = hook(enc, inputs, targets, 2)
    assert loss_late.ndim == 0 and loss_late.item() > 0


def test_train_image_mixup_disabled(tmp_path: Path) -> None:
    """mixup_alpha=0 disables Mixup (no batch_hook) and still trains end-to-end."""
    from memo.training.train_image import run_train_image

    def _fake_image_loader(_path: str) -> Tensor:
        return torch.randn(3, 112, 112)

    n = 32
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(
        data_dir / "train.csv",
        [{"path": f"{i}.png", "label": i % NUM_CLASSES} for i in range(n)],
        ["path", "label"],
    )

    ckpt = tmp_path / "image_nomix.pt"
    run_train_image(
        data_dir,
        epochs=4,
        out=ckpt,
        remap_from="ekman7",
        mixup_alpha=0.0,
        loader=_fake_image_loader,
        encoder=_StubEncoder("image", input_dim=3 * 112 * 112),
        runs_dir=tmp_path / "runs",
    )
    _assert_run_produced_artifacts(tmp_path / "runs", ckpt)
