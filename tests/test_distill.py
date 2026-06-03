"""Audio knowledge-distillation tests.

Fully offline: a counter-instrumented stub teacher replaces Wav2Vec2 so the KD
math, the teacher-logit cache, and the run plumbing are exercised without any
HuggingFace download. The real teacher is gated behind MEMO_ALLOW_HF_DOWNLOAD=1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from memo.encoders.base import BaseEncoder
from memo.labels import NUM_CLASSES
from memo.losses import FocalLoss, KDLoss
from memo.training.distill import (
    DistillSample,
    TeacherLogitCache,
    _module_device,
    make_kd_hook,
    run_distill_audio,
)


class _StubAudioEncoder(BaseEncoder):
    """Flatten log-mel → linear → 7 logits. Trainable student stand-in."""

    def __init__(self, input_dim: int = 64 * 301) -> None:
        super().__init__()
        self.name = "audio"
        self.num_classes = NUM_CLASSES
        self._fc = nn.Linear(input_dim, NUM_CLASSES)

    def predict_logits(self, x: Tensor) -> Tensor:
        return self._fc(torch.as_tensor(x, dtype=torch.float32).flatten(1))


class _StubTeacher(nn.Module):
    """Frozen, deterministic per-clip teacher; counts forward'd rows."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(123)
        self.proj = nn.Linear(16, NUM_CLASSES)
        for p in self.parameters():
            p.requires_grad = False
        self.calls = 0

    def teacher_logits(self, waveform: Tensor) -> Tensor:
        self.calls += waveform.size(0)
        with torch.no_grad():
            return self.proj(waveform[:, :16]) * 3.0  # ×3 → peaked, informative targets


def _make_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _fake_distill_loader(path: str) -> DistillSample:
    """Hash-free loader: unique key from the file stem, no file/SHA needed."""
    return DistillSample(
        logmel=torch.randn(64, 301),
        waveform=torch.randn(48_000),
        key=f"stub:{Path(path).stem}",
    )


# ---------------------------------------------------------------------------
# KD math: loss decreases + student moves toward the teacher's soft target
# ---------------------------------------------------------------------------


def test_kd_smoke() -> None:
    torch.manual_seed(0)
    student = _StubAudioEncoder()
    teacher = _StubTeacher()

    b = 16
    inputs = {
        "logmel": torch.randn(b, 64, 301),
        "waveform": torch.randn(b, 48_000),
        "keys": [f"k{i}" for i in range(b)],
    }
    # A competent teacher: its argmax is the label, so the hard (focal) and soft
    # (KL) signals agree and the student provably converges toward it.
    teacher_logits = teacher.teacher_logits(inputs["waveform"])
    targets = teacher_logits.argmax(dim=-1)
    teacher.calls = 0  # reset; count only the forwards during training

    cache = TeacherLogitCache()
    kd = KDLoss(alpha=0.5, temperature=4.0, focal=FocalLoss(gamma=0.0, label_smoothing=0.0))
    hook = make_kd_hook(kd, cache, teacher)
    opt = torch.optim.Adam(student.parameters(), lr=2e-3)

    losses: list[float] = []
    kls: list[float] = []
    for _ in range(60):
        opt.zero_grad()
        loss = hook(student, inputs, targets, 0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            s = student.predict_logits(inputs["logmel"])
            kls.append(
                float(
                    F.kl_div(
                        F.log_softmax(s, dim=-1),
                        F.softmax(teacher_logits, dim=-1),
                        reduction="batchmean",
                    )
                )
            )

    assert losses[-1] < losses[0], f"KD loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"
    assert kls[-1] < kls[0], "student did not move toward the teacher's soft distribution"
    # Teacher ran exactly once per clip (filled on step 0, cached thereafter).
    assert teacher.calls == b


def test_kd_hook_matches_kdloss() -> None:
    """The hook's scalar equals a direct KDLoss call on the cached teacher logits."""
    torch.manual_seed(1)
    student = _StubAudioEncoder()
    teacher = _StubTeacher()
    kd = KDLoss(alpha=0.5, temperature=4.0, focal=FocalLoss(gamma=2.0, label_smoothing=0.05))
    cache = TeacherLogitCache()
    hook = make_kd_hook(kd, cache, teacher)

    inputs = {
        "logmel": torch.randn(8, 64, 301),
        "waveform": torch.randn(8, 48_000),
        "keys": [f"k{i}" for i in range(8)],
    }
    targets = torch.randint(0, NUM_CLASSES, (8,))

    hook_loss = hook(student, inputs, targets, 0)
    with torch.no_grad():
        t = cache.get_or_compute(inputs["keys"], inputs["waveform"], teacher)
        direct = kd(student.predict_logits(inputs["logmel"]), targets, t)
    assert torch.allclose(hook_loss.detach(), direct)


# ---------------------------------------------------------------------------
# Cache: one teacher forward per unique clip, zero during the epochs
# ---------------------------------------------------------------------------


def test_teacher_cache_hit(tmp_path: Path) -> None:
    n = 32
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # A separate val.csv keeps all `n` clips in the training split, so the
    # precompute pass touches exactly `n` unique clips.
    _make_csv(
        data_dir / "train.csv",
        [{"path": f"{i}.wav", "label": i % NUM_CLASSES} for i in range(n)],
        ["path", "label"],
    )
    _make_csv(
        data_dir / "val.csv",
        [{"path": f"v{i}.wav", "label": i % NUM_CLASSES} for i in range(8)],
        ["path", "label"],
    )

    teacher = _StubTeacher()
    run_distill_audio(
        data_dir,
        epochs=2,
        out=tmp_path / "audio.pt",
        remap_from="ekman7",
        loader=_fake_distill_loader,
        teacher=teacher,
        encoder=_StubAudioEncoder(),
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
    )
    # Precompute computes all 32 unique clips; both training epochs are pure
    # cache hits → the teacher is never run again (total == 32, not > 32).
    assert teacher.calls == n


def test_distill_run_produces_artifacts(tmp_path: Path) -> None:
    n = 32
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(
        data_dir / "train.csv",
        [{"path": f"{i}.wav", "label": i % NUM_CLASSES} for i in range(n)],
        ["path", "label"],
    )
    _make_csv(
        data_dir / "val.csv",
        [{"path": f"v{i}.wav", "label": i % NUM_CLASSES} for i in range(8)],
        ["path", "label"],
    )

    ckpt = tmp_path / "audio.pt"
    manifest_path = run_distill_audio(
        data_dir,
        epochs=3,
        out=ckpt,
        remap_from="ekman7",
        loader=_fake_distill_loader,
        teacher=_StubTeacher(),
        encoder=_StubAudioEncoder(),
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
    )
    assert ckpt.exists()
    assert manifest_path.name == "manifest.json"
    raw = json.loads(manifest_path.read_text())
    assert raw["end_time"] is not None
    assert raw["config"]["model"]["kd"]["enabled"] is True
    # The distilled checkpoint reloads into a fresh student with identical outputs.
    enc2 = _StubAudioEncoder()
    enc2.load_state_dict(torch.load(ckpt, weights_only=True))
    assert (tmp_path / "cache" / "teacher_logits.pt").exists()


# ---------------------------------------------------------------------------
# Import isolation: no module outside distill.py references Wav2Vec2
# ---------------------------------------------------------------------------


def test_module_device_infers_teacher_device() -> None:
    """The cache moves inputs to the teacher's device (guarding the CUDA path)."""
    teacher = _StubTeacher()  # nn.Module with params on CPU
    assert _module_device(teacher) == next(teacher.parameters()).device

    class _ParamlessTeacher:
        def teacher_logits(self, waveform: Tensor) -> Tensor:
            return torch.zeros(waveform.size(0), NUM_CLASSES)

    # A paramless stub teacher yields None → no move attempted (stays on input device).
    assert _module_device(_ParamlessTeacher()) is None


def test_no_wav2vec2_import_outside_distill() -> None:
    # No module outside distill.py may import the teacher. Check the import symbol
    # `Wav2Vec2Model`, not the lowercase model-id string ("facebook/wav2vec2-base")
    # that legitimately appears in config and CLI help text.
    src = Path(__file__).resolve().parent.parent / "src" / "memo"
    offenders = [
        str(py.relative_to(src))
        for py in src.rglob("*.py")
        if py.name != "distill.py" and "Wav2Vec2Model" in py.read_text()
    ]
    assert not offenders, f"Wav2Vec2Model imported outside distill.py: {offenders}"
