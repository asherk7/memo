"""Audio knowledge distillation.

A frozen Wav2Vec2-Base teacher distills into the 0.5M-param `LogMelCRNNEncoder`
student via ``memo train audio --distill``. The teacher's soft targets are
precomputed once per training set and cached to disk, so Wav2Vec2 is never
re-run epoch over epoch. The teacher exists only at train time; the inference
graph stays the CRNN. The real teacher is gated behind
``MEMO_ALLOW_HF_DOWNLOAD=1``; tests inject a stub teacher and stay offline.

Wav2Vec2 consumes the raw 16 kHz waveform, the CRNN consumes the log-mel, so the
distill data path carries both (a `DistillSample`), keyed by an opaque per-clip
string so the cache survives the class-balanced sampler's reshuffling.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from ..encoders.audio import LogMelCRNNEncoder
from ..encoders.base import BaseEncoder
from ..labels import NUM_CLASSES, EkmanEmotion, remap_cremad, remap_ravdess
from ..losses import KDLoss
from ..preprocessing.audio import (
    SAMPLE_RATE,
    fix_length,
    log_mel_spectrogram,
    resample,
)
from ..seed import seed_everything
from .datasets import CsvDataset, focal_loss_from_labels, stratified_train_val_split
from .manifest import RunManifest, new_run_id
from .samplers import ClassBalancedSampler
from .trainer import BatchHook, Trainer

if TYPE_CHECKING:
    from ..config import ExperimentConfig, KDConfig

__all__ = [
    "TeacherProtocol",
    "Wav2Vec2EmotionTeacher",
    "build_teacher",
    "fit_teacher_probe",
    "DistillSample",
    "distill_collate",
    "distill_val_collate",
    "TeacherLogitCache",
    "precompute_teacher_logits",
    "make_kd_hook",
    "run_distill_audio",
]


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------


@runtime_checkable
class TeacherProtocol(Protocol):
    """Anything that turns a raw-waveform batch into ``(B, 7)`` emotion logits."""

    def teacher_logits(self, waveform: torch.Tensor) -> torch.Tensor: ...


class Wav2Vec2EmotionTeacher(nn.Module):
    """Frozen wav2vec2-base SSL features + a linear emotion probe (768 → 7).

    ``facebook/wav2vec2-base`` is self-supervised and has no emotion head, so a
    cheap linear probe over its frozen mean-pooled features supplies informative
    7-class soft targets without backpropagating through the 95M-param backbone.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base") -> None:
        super().__init__()
        from transformers import Wav2Vec2Model  # isolated local import

        model = Wav2Vec2Model.from_pretrained(model_name).eval()
        for p in model.parameters():
            p.requires_grad = False
        hidden = int(model.config.hidden_size)  # 768
        self.backbone = model
        self.probe = nn.Linear(hidden, NUM_CLASSES)

    @torch.no_grad()
    def features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mean-pooled frozen features, ``(B, S) → (B, 768)``.

        Per-utterance zero-mean/unit-variance normalization matches what the
        Wav2Vec2 feature extractor applies with ``do_normalize=True``.
        """
        x = (waveform - waveform.mean(dim=-1, keepdim=True)) / (
            waveform.std(dim=-1, keepdim=True) + 1e-7
        )
        return self.backbone(x).last_hidden_state.mean(dim=1)

    def teacher_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.probe(self.features(waveform))


def fit_teacher_probe(
    teacher: Wav2Vec2EmotionTeacher,
    waveforms: list[torch.Tensor],
    labels: list[int],
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
) -> None:
    """Fit ``teacher.probe`` on frozen mean-pooled features (backbone stays frozen).

    Runs only on the real-teacher path; the stub-teacher tests inject a ready
    teacher and never call this.
    """
    feats = torch.cat([teacher.features(w.unsqueeze(0).to(device)) for w in waveforms]).detach()
    y = torch.tensor(labels, dtype=torch.long, device=device)
    probe = teacher.probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(feats), y)
        loss.backward()
        opt.step()


def build_teacher(cfg_kd: KDConfig, device: str = "cpu") -> TeacherProtocol:
    """Construct the real Wav2Vec2 teacher. Gated behind ``MEMO_ALLOW_HF_DOWNLOAD=1``.

    Tests inject a stub teacher and never call this.
    """
    import os

    if os.environ.get("MEMO_ALLOW_HF_DOWNLOAD") != "1":
        raise RuntimeError(
            "The real Wav2Vec2 teacher requires MEMO_ALLOW_HF_DOWNLOAD=1 (a ~360 MB "
            "HuggingFace download). Tests must inject a stub teacher instead."
        )
    return Wav2Vec2EmotionTeacher(cfg_kd.teacher).to(device)


# ---------------------------------------------------------------------------
# Dual-view samples (student log-mel + teacher waveform) and collation
# ---------------------------------------------------------------------------


@dataclass
class DistillSample:
    """One clip in both representations the two networks need.

    ``logmel`` (64, T) feeds the student; ``waveform`` (S,) feeds the teacher
    (used only on a cache miss); ``key`` keys the teacher-logit cache.
    """

    logmel: torch.Tensor
    waveform: torch.Tensor
    key: str


def distill_collate(
    batch: list[tuple[DistillSample, int]],
) -> tuple[dict[str, object], torch.Tensor]:
    """Train collate → ``({"logmel", "waveform", "keys"}, labels)``.

    Tensor values move to the device under `Trainer._to_device`; the ``keys``
    list passes through untouched.
    """
    samples, labels = zip(*batch, strict=False)
    return (
        {
            "logmel": torch.stack([s.logmel for s in samples]),
            "waveform": torch.stack([s.waveform for s in samples]),
            "keys": [s.key for s in samples],
        },
        torch.tensor(labels, dtype=torch.long),
    )


def distill_val_collate(
    batch: list[tuple[DistillSample, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Val collate → ``(logmel_batch, labels)``.

    Validation evaluates the student with no teacher in the loop, so `_evaluate`
    gets a plain log-mel tensor, not the train-time dict batch.
    """
    samples, labels = zip(*batch, strict=False)
    return (
        torch.stack([s.logmel for s in samples]),
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Teacher-logit cache
# ---------------------------------------------------------------------------


def _module_device(obj: object) -> torch.device | None:
    """The device of an `nn.Module`'s first parameter, or ``None`` (e.g. a
    paramless stub teacher) — used to keep cache inputs on the teacher's device."""
    if isinstance(obj, nn.Module):
        params = list(obj.parameters())
        if params:
            return params[0].device
    return None


class TeacherLogitCache:
    """Opaque-key → ``(7,)`` teacher logits, in-memory with optional disk persistence.

    ``misses`` counts cache-miss rows (teacher forwards), so tests can assert the
    teacher runs exactly once per unique clip and zero times thereafter.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.misses = 0
        if path is not None and path.exists():
            loaded: dict[str, torch.Tensor] = torch.load(path, weights_only=True)
            self._mem = loaded
        else:
            self._mem = {}

    def get_or_compute(
        self, keys: list[str], waveform: torch.Tensor, teacher: TeacherProtocol
    ) -> torch.Tensor:
        """Return ``(B, 7)`` teacher logits for ``keys``, computing only misses.

        The teacher forward runs once for the unique uncached rows in the batch;
        cached keys are served from memory. Returned tensors are CPU detached
        clones — the caller moves them to the student's device.
        """
        missing = [(i, k) for i, k in enumerate(keys) if k not in self._mem]
        if missing:
            self.misses += len(missing)
            rows = torch.tensor([i for i, _ in missing])
            batch = waveform[rows]
            # Move to the teacher's device: the precompute pass builds a plain
            # CPU DataLoader, so a CUDA teacher would otherwise see a CPU input.
            dev = _module_device(teacher)
            if dev is not None and batch.device != dev:
                batch = batch.to(dev)
            computed = teacher.teacher_logits(batch).detach().cpu()
            for out_row, (_, k) in enumerate(missing):
                self._mem[k] = computed[out_row].clone()
        return torch.stack([self._mem[k] for k in keys])

    def flush(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._mem, self.path)


def precompute_teacher_logits(
    dataset: object,
    teacher: TeacherProtocol,
    cache: TeacherLogitCache,
    *,
    batch_size: int,
) -> int:
    """Fill ``cache`` for every clip in ``dataset`` in one ordered pass.

    Returns the number of unique clips cached. The ordered, no-sampler pass makes
    the teacher-forward count deterministic (one per unique clip), independent of
    the class-balanced sampler's with-replacement draws, and guarantees zero
    teacher forwards during the epochs themselves.
    """
    loader: DataLoader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        collate_fn=distill_collate,
    )
    before = len(cache._mem)
    for inputs, _ in loader:
        cache.get_or_compute(inputs["keys"], inputs["waveform"], teacher)
    return len(cache._mem) - before


# ---------------------------------------------------------------------------
# KD training hook + entry point
# ---------------------------------------------------------------------------


def make_kd_hook(kd_loss: KDLoss, cache: TeacherLogitCache, teacher: TeacherProtocol) -> BatchHook:
    """Build a `Trainer.batch_hook` computing the KD loss from cached teacher logits.

    After `precompute_teacher_logits`, ``cache.get_or_compute`` is a pure lookup;
    the teacher never runs during the training epochs.
    """

    def hook(model: BaseEncoder, inputs: object, targets: torch.Tensor, epoch: int) -> torch.Tensor:
        assert isinstance(inputs, dict)
        student = model.predict_logits(inputs["logmel"])
        teacher_logits = cache.get_or_compute(inputs["keys"], inputs["waveform"], teacher).to(
            student.device
        )
        return kd_loss(student, targets, teacher_logits)

    return hook


_REMAPPERS: dict[str, Callable] = {
    "ravdess": remap_ravdess,
    "cremad": remap_cremad,
    "ekman7": lambda x: EkmanEmotion(int(x)),
}


def _make_distill_loader(dataset_id: str, is_train: bool) -> Callable[[str], DistillSample]:
    """Real distill loader: WAV → clean 16 kHz waveform (teacher) + log-mel (student).

    The waveform stays clean (the teacher's cached target must be deterministic);
    only the student's log-mel is SpecAugmented at train time. The cache key is
    content-addressed — ``(dataset_id, file stem, sha256(bytes))`` — so it is
    stable across runs and detects a changed file.
    """
    from ..augment.audio import spec_augment

    def _load(path: str) -> DistillSample:
        import io

        import soundfile as sf

        # Read once: the bytes feed both the content-addressed cache key and the
        # decoder (via an in-memory buffer), avoiding a second disk read.
        raw = Path(path).read_bytes()
        key = f"{dataset_id}:{Path(path).stem}:{hashlib.sha256(raw).hexdigest()[:16]}"
        waveform_np, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)
        clean = fix_length(resample(waveform_np, sr, SAMPLE_RATE))
        logmel = torch.from_numpy(log_mel_spectrogram(clean, SAMPLE_RATE))
        if is_train:
            logmel = spec_augment(logmel)
        return DistillSample(logmel=logmel, waveform=torch.from_numpy(clean), key=key)

    return _load


def run_distill_audio(
    data_dir: Path,
    *,
    epochs: int = 15,
    out: Path,
    config: ExperimentConfig | None = None,
    device: str = "cpu",
    runs_dir: Path = Path("runs"),
    remap_from: str = "ravdess",
    val_split: float = 0.1,
    dataset_id: str = "ravdess",
    loader: Callable[[str], DistillSample] | None = None,
    teacher: TeacherProtocol | None = None,
    encoder: BaseEncoder | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Distill a frozen Wav2Vec2-Base teacher into the CRNN student.

    Args mirror `run_train_audio`, plus:
        dataset_id: tag baked into the content-addressed cache key.
        loader: custom loader returning a `DistillSample` (tests inject a
            hash-free stub so no file is read and no SHA is computed).
        teacher: a `TeacherProtocol`; ``None`` builds the real Wav2Vec2 teacher
            (HF-gated) and fits its linear probe on the training clips.
        encoder: the student; ``None`` builds a fresh `LogMelCRNNEncoder`.
        cache_dir: where teacher logits persist; defaults to the run dir.

    Returns the path to the written ``manifest.json``.
    """
    from ..config import ExperimentConfig

    cfg = config or ExperimentConfig()
    cfg.train.epochs = epochs
    cfg.model.kd.enabled = True
    seed_everything(cfg.seed)

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    run_id = new_run_id("distill")
    run_dir = Path(runs_dir) / run_id
    manifest = RunManifest.create(run_id, cfg, [str(data_dir)], cfg.seed)
    logger.info("audio distillation run {} → {}", run_id, run_dir)

    # dual-view dataset: student log-mel + teacher waveform
    train_loader_fn = loader if loader is not None else _make_distill_loader(dataset_id, True)
    val_loader_fn = loader if loader is not None else _make_distill_loader(dataset_id, False)

    train_csv = Path(data_dir) / "train.csv"
    val_csv = Path(data_dir) / "val.csv"
    full_ds = CsvDataset(train_csv, loader=train_loader_fn, remap=remap, root=data_dir)

    if val_csv.exists():
        val_ds: object = CsvDataset(val_csv, loader=val_loader_fn, remap=remap, root=data_dir)
        train_sub: object = full_ds
        train_labels = full_ds.labels
    else:
        train_sub, train_labels, val_ds, _ = stratified_train_val_split(
            full_ds, val_split, cfg.seed
        )

    sampler = ClassBalancedSampler(
        train_labels,
        beta=cfg.train.focal_loss.class_weight_beta,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_dl: DataLoader = DataLoader(
        train_sub,  # type: ignore[arg-type]
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        collate_fn=distill_collate,
    )
    val_dl: DataLoader = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=cfg.train.batch_size * 2,
        collate_fn=distill_val_collate,
    )

    # teacher (+ probe fit on the real path)
    if teacher is None:
        teacher = build_teacher(cfg.model.kd, device)
        if isinstance(teacher, Wav2Vec2EmotionTeacher):
            waveforms = [full_ds[i][0].waveform for i in range(len(full_ds))]
            fit_teacher_probe(
                teacher, waveforms, full_ds.labels, epochs=cfg.model.kd.probe_epochs, device=device
            )

    # precompute teacher logits once (deterministic cache)
    cache_root = cache_dir if cache_dir is not None else run_dir
    cache = TeacherLogitCache(Path(cache_root) / "teacher_logits.pt")
    n_cached = precompute_teacher_logits(train_sub, teacher, cache, batch_size=cfg.train.batch_size)
    logger.info("precomputed {} teacher logits ({} forwards)", n_cached, cache.misses)

    # student + KD loss
    enc = (
        encoder
        if encoder is not None
        else LogMelCRNNEncoder(n_mels=cfg.model.encoders.audio.n_mels)
    )
    focal = focal_loss_from_labels(train_labels, cfg)
    kd_loss = KDLoss(alpha=cfg.model.kd.alpha, temperature=cfg.model.kd.temperature, focal=focal)
    hook = make_kd_hook(kd_loss, cache, teacher)

    trainer = Trainer(
        enc,
        focal,  # placeholder LossFn; the batch_hook owns the real KD loss
        cfg.train,
        max_lr=cfg.train.scheduler.max_lr.audio,
        device=device,
        batch_hook=hook,
    )

    result = trainer.fit(train_dl, val_dl)
    cache.flush()
    logger.info("distillation complete: best_val_macro_f1={:.4f}", result.best_metric or 0.0)

    out = Path(out)
    trainer.save_checkpoint(out)
    manifest.finalize(metrics={"best_val_macro_f1": result.best_metric or 0.0})
    return manifest.write(run_dir)
