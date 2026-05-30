"""Stage-2 optional joint multimodal fine-tune (§4.2, Phase 10).

``memo train joint`` starts from the stage-1 checkpoints, unfreezes a small slice
of each encoder (the last MobileNet block, the last 2 MiniLM layers or the LoRA
adapters, and the full audio CRNN), and fine-tunes all three together under
per-sample modality dropout with a multi-task loss:

    L = L_fused + Σ_i λ_i · L_i   (λ_i = 0.3)

`L_fused` is the label-smoothed NLL of the confidence-gated fused distribution;
each `L_i` is the stage-1 FocalLoss on a single encoder's logits, supervised only
on the rows where that modality survived dropout — so every encoder keeps a
direct training signal even when the fused loss is carried by another modality.

The 7 fusion scalars are **frozen** here (excluded from the optimizer); they get
their own calibration in Phase 11. The stage is skippable — Phase 11 calibration
runs with or without it. The checkpoint is written per-encoder so it reloads
through `MultimodalEmotionPipeline.from_config` with no code change.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..encoders.audio import LogMelCRNNEncoder
from ..encoders.base import BaseEncoder
from ..encoders.image import MobileNetV3SmallFaceEncoder
from ..encoders.text import MiniLMTextEncoder
from ..fusion import LateFusion
from ..labels import EkmanEmotion, remap_goemotions
from ..losses import FocalLoss
from ..seed import seed_everything
from .datasets import JsonlDataset, focal_loss_from_labels
from .manifest import RunManifest, new_run_id
from .modality_dropout import modality_keep_mask
from .trainer import build_param_groups

__all__ = ["fused_nll", "joint_loss", "run_train_joint"]

MODALITIES: tuple[str, ...] = ("image", "text", "audio")

_REMAPPERS: dict[str, Callable[[Any], EkmanEmotion]] = {
    "goemotions": remap_goemotions,
    "ekman7": lambda x: EkmanEmotion(int(x)),
}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def fused_nll(
    probs: torch.Tensor, targets: torch.Tensor, *, label_smoothing: float = 0.05
) -> torch.Tensor:
    """Label-smoothed NLL on a fused probability distribution.

    `LateFusion` returns probabilities (already softmaxed + mixed), so `FocalLoss`
    — which re-applies ``log_softmax`` — cannot consume them. This computes NLL in
    log-space against the same smoothed targets the stage-1 loss uses, with a
    ``clamp_min`` floor guarding ``log(0)``.
    """
    log_p = probs.clamp_min(1e-8).log()
    k = probs.size(-1)
    q = torch.full_like(log_p, label_smoothing / k)
    q.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / k)
    return -(q * log_p).sum(dim=-1).mean()


def joint_loss(
    fused_probs: torch.Tensor,
    per_modality_logits: dict[str, torch.Tensor],
    keep_mask: dict[str, torch.Tensor],
    targets: torch.Tensor,
    focal_none: FocalLoss,
    *,
    lam: float = 0.3,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task loss ``L = L_fused + Σ λ_i L_i`` (§4.2).

    Aux losses are **indexed** to kept rows (``per_sample[kept].mean()``), not
    multiplied by zero — supervising a dropped row would teach the head to map an
    unused input to the label. The ``logits.sum()*0`` fallback keeps the graph
    connected on the (vanishingly rare) batch where a modality is dropped for
    every sample.
    """
    loss = fused_nll(fused_probs, targets, label_smoothing=label_smoothing)
    aux: dict[str, torch.Tensor] = {}
    for m, logits in per_modality_logits.items():
        per_sample = focal_none(logits, targets)  # (B,)
        kept = keep_mask[m]
        aux[m] = per_sample[kept].mean() if bool(kept.any()) else logits.sum() * 0.0
        loss = loss + lam * aux[m]
    return loss, aux


# ---------------------------------------------------------------------------
# Per-encoder unfreeze (§4.2) — defensive: a no-op on stub encoders in tests
# ---------------------------------------------------------------------------


def _unfreeze_image(enc: BaseEncoder) -> None:
    """Freeze the MobileNet backbone, then unfreeze its last block + the head."""
    backbone = getattr(enc, "backbone", None)
    if backbone is None:  # stub / unexpected layout → leave fully trainable
        return
    features = getattr(backbone, "features", None)
    if features is None:
        return
    for p in backbone.parameters():
        p.requires_grad = False
    for p in features[-1].parameters():
        p.requires_grad = True
    _unfreeze_head(enc)


def _unfreeze_text(enc: BaseEncoder, *, lora: bool) -> None:
    """Unfreeze the head; for non-LoRA also the last 2 MiniLM transformer layers.

    With LoRA, peft already leaves only the adapters trainable, so we touch only
    the head.
    """
    _unfreeze_head(enc)
    if lora:
        return
    backbone = getattr(enc, "backbone", None)
    if backbone is None:  # stub layout → head already unfrozen above
        return
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is None:
        return
    for p in backbone.parameters():
        p.requires_grad = False
    for idx in (len(layers) - 2, len(layers) - 1):
        for p in layers[idx].parameters():
            p.requires_grad = True


def _unfreeze_audio(enc: BaseEncoder) -> None:
    """The full CRNN trains (it has no frozen backbone)."""
    for p in enc.parameters():
        p.requires_grad = True


def _unfreeze_head(enc: BaseEncoder) -> None:
    head = getattr(enc, "head", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True


def _unfreeze_for_joint(encoders: dict[str, BaseEncoder], *, lora: bool) -> None:
    _unfreeze_image(encoders["image"])
    _unfreeze_text(encoders["text"], lora=lora)
    _unfreeze_audio(encoders["audio"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _default_joint_collate(
    batch: list[tuple[dict[str, Any], int]],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Stack an aligned multimodal batch.

    Each modality must be present (non-``None``) for every row or absent for every
    row — mixed presence within a batch is a misaligned manifest and raises.
    Image/audio values are stacked tensors; text is batch-tokenized when it
    arrives as raw strings, or stacked when already a token dict.
    """
    from ..preprocessing.text import preprocess_text

    dicts, labels = zip(*batch, strict=False)
    out: dict[str, Any] = {}
    for m in MODALITIES:
        vals = [d.get(m) for d in dicts]
        present = [v is not None for v in vals]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"Modality {m!r} is present for only some rows in the batch.")
        first = vals[0]
        if m == "text" and isinstance(first, str):
            out[m] = preprocess_text(list(vals))
        elif isinstance(first, dict):
            out[m] = {k: torch.stack([v[k] for v in vals]) for k in first}
        else:
            out[m] = torch.stack(vals)
    return out, torch.tensor(labels, dtype=torch.long)


def _default_joint_loaders() -> dict[str, Callable[[Any], Any]]:
    """Real per-modality loaders: image/audio paths → tensors, text → raw string."""
    from ..preprocessing.audio import preprocess_audio
    from ..preprocessing.face import FaceNotFoundError, preprocess_face

    def _img(path: str) -> torch.Tensor:
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            return preprocess_face(img)
        except FaceNotFoundError:
            return torch.zeros(3, 112, 112)

    def _aud(path: str) -> torch.Tensor:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return preprocess_audio(wav, sr)

    return {"image": _img, "text": lambda s: s, "audio": _aud}


# ---------------------------------------------------------------------------
# Forward / device helpers
# ---------------------------------------------------------------------------


def _move(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[m] = v.to(device)
        elif isinstance(v, dict):
            out[m] = {k: t.to(device) for k, t in v.items()}
        else:
            out[m] = v
    return out


def _encode_present(
    encoders: dict[str, BaseEncoder], inputs: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Run each present modality's encoder → ``{modality: (B, 7) logits}``."""
    return {m: encoders[m].predict_logits(inputs[m]) for m in MODALITIES if m in inputs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_ckpt(enc: BaseEncoder, ckpt: Path | None, device: str) -> None:
    if ckpt is not None:
        enc.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))


def run_train_joint(
    aligned_train: Path,
    aligned_val: Path,
    *,
    out: Path,
    epochs: int = 8,
    config: ExperimentConfig | None = None,
    device: str = "cpu",
    runs_dir: Path = Path("runs"),
    remap_from: str = "ekman7",
    aux_lambda: float | None = None,
    lora: bool = False,
    image_ckpt: Path | None = None,
    text_ckpt: Path | None = None,
    audio_ckpt: Path | None = None,
    loaders: dict[str, Callable[[Any], Any]] | None = None,
    collate_fn: Callable[[list[tuple[dict[str, Any], int]]], tuple[dict[str, Any], torch.Tensor]]
    | None = None,
    encoders: dict[str, BaseEncoder] | None = None,
    fusion: LateFusion | None = None,
) -> Path:
    """Joint multimodal fine-tune (§4.2). Writes one checkpoint per encoder.

    Args:
        aligned_train / aligned_val: aligned multimodal JSONL paths (Phase 7 schema).
        out: checkpoint stem; per-encoder files are written as ``<stem>_image.pt``
            etc., the slots `from_config` reads.
        aux_lambda: overrides ``config.train.joint.aux_lambda`` when given.
        lora: the text encoder uses LoRA adapters (changes the unfreeze policy).
        image_ckpt / text_ckpt / audio_ckpt: stage-1 checkpoints to warm-start.
        loaders / collate_fn / encoders / fusion: injection points for offline
            tests; ``None`` builds the real components.

    Returns the path to the written ``manifest.json``.
    """
    cfg = config or ExperimentConfig()
    cfg.train.epochs = epochs
    if aux_lambda is not None:
        cfg.train.joint.aux_lambda = aux_lambda
    seed_everything(cfg.seed)

    remap = _REMAPPERS.get(remap_from)
    if remap is None:
        raise ValueError(f"Unknown remap_from={remap_from!r}; choose from {list(_REMAPPERS)}")

    dev = torch.device(device)
    run_id = new_run_id("joint")
    run_dir = Path(runs_dir) / run_id
    manifest = RunManifest.create(run_id, cfg, [str(aligned_train), str(aligned_val)], cfg.seed)
    logger.info("joint fine-tune run {} → {}", run_id, run_dir)

    # ---- encoders (warm-started from stage 1) + frozen fusion -----------
    if encoders is None:
        encoders = {
            "image": MobileNetV3SmallFaceEncoder(pretrained=True),
            "text": MiniLMTextEncoder(lora=lora, dropout=cfg.model.encoders.text.head_dropout),
            "audio": LogMelCRNNEncoder(n_mels=cfg.model.encoders.audio.n_mels),
        }
        _load_ckpt(encoders["image"], image_ckpt, device)
        _load_ckpt(encoders["text"], text_ckpt, device)
        _load_ckpt(encoders["audio"], audio_ckpt, device)

    _unfreeze_for_joint(encoders, lora=lora)
    for enc in encoders.values():
        if isinstance(enc, nn.Module):
            enc.to(dev).train()

    fusion = fusion if fusion is not None else LateFusion()
    fusion.to(dev)
    for p in fusion.parameters():  # fusion scalars frozen — calibrated in Phase 11
        p.requires_grad = False

    # ---- data -----------------------------------------------------------
    mod_loaders = loaders if loaders is not None else _default_joint_loaders()
    collate = collate_fn if collate_fn is not None else _default_joint_collate
    train_ds = JsonlDataset(aligned_train, loaders=mod_loaders, remap=remap)
    val_ds = JsonlDataset(aligned_val, loaders=mod_loaders, remap=remap)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.train.batch_size * 2, shuffle=False, collate_fn=collate
    )

    # ---- optimizer: three groups (backbones 1e-5, heads 1e-4), fusion frozen
    groups: list[dict[str, Any]] = []
    for enc in encoders.values():
        groups.extend(build_param_groups(enc, cfg.train.joint.backbone_lr, cfg.train.joint.head_lr))
    if not groups:
        raise ValueError("No trainable parameters across the encoders for joint fine-tune.")
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        epochs=epochs,
        steps_per_epoch=len(train_dl),
    )

    focal_none = focal_loss_from_labels(train_ds.labels, cfg)
    focal_none.reduction = "none"  # per-sample aux losses, masked to kept rows
    gen = torch.Generator().manual_seed(cfg.seed)
    trainable = [p for g in groups for p in g["params"]]

    epoch_losses: list[float] = []
    best_f1 = -1.0
    best_states: dict[str, dict[str, torch.Tensor]] | None = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        for enc in encoders.values():
            enc.train()
        running, n_batches = 0.0, 0
        for inputs, targets in train_dl:
            inputs = _move(inputs, dev)
            targets = targets.to(dev)
            present = [m for m in MODALITIES if m in inputs]
            keep = modality_keep_mask(
                targets.size(0),
                present,
                rate=cfg.train.modality_dropout.rate,
                text_rate=cfg.train.modality_dropout.text_rate,
                generator=gen,
            )
            keep = {m: t.to(dev) for m, t in keep.items()}

            logits = _encode_present(encoders, inputs)
            fused = fusion.fuse(logits, keep_mask=keep).probs
            loss, _ = joint_loss(
                fused,
                logits,
                keep,
                targets,
                focal_none,
                lam=cfg.train.joint.aux_lambda,
                label_smoothing=cfg.train.focal_loss.label_smoothing,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach())
            n_batches += 1
        epoch_losses.append(running / max(n_batches, 1))

        macro_f1 = _evaluate_fused(encoders, fusion, val_dl, dev)
        logger.info(
            "joint epoch {} loss={:.4f} val_fused_macro_f1={:.4f}",
            epoch,
            epoch_losses[-1],
            macro_f1,
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_states = {m: copy.deepcopy(enc.state_dict()) for m, enc in encoders.items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.train.early_stopping_patience:
                break

    if best_states is not None:
        for m, enc in encoders.items():
            enc.load_state_dict(best_states[m])

    # ---- save per-encoder checkpoints (reload via from_config) ----------
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for m, enc in encoders.items():
        torch.save(enc.state_dict(), out.with_name(f"{out.stem}_{m}.pt"))

    manifest.finalize(metrics={"fused_macro_f1": best_f1, "final_train_loss": epoch_losses[-1]})
    logger.info("joint fine-tune complete: best fused macro-F1={:.4f}", best_f1)
    return manifest.write(run_dir)


@torch.no_grad()
def _evaluate_fused(
    encoders: dict[str, BaseEncoder],
    fusion: LateFusion,
    val_dl: DataLoader,
    device: torch.device,
) -> float:
    """Fused macro-F1 on the val set with all modalities present (no dropout)."""
    for enc in encoders.values():
        enc.eval()
    preds: list[int] = []
    trues: list[int] = []
    for inputs, targets in val_dl:
        inputs = _move(inputs, device)
        logits = _encode_present(encoders, inputs)
        fused = fusion.fuse(logits).probs
        preds.extend(fused.argmax(dim=-1).cpu().tolist())
        trues.extend(targets.tolist())
    return float(f1_score(trues, preds, average="macro", zero_division=0))
