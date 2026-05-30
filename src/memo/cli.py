"""MEMO command-line interface (§8).

Entry point registered as ``memo`` in ``pyproject.toml``.  Subcommands are
wired per-phase; later phases add their own commands to ``train_app`` and the
top-level ``app`` directly.

Currently wired (Phase 8):
    memo train image  --data <dir> --epochs N --out <ckpt>
    memo train text   --data <dir> --epochs N [--lora] --out <ckpt>
    memo train audio  --data <dir> --epochs N [--k-fold] --out <ckpt>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="memo",
    help="MEMO multimodal emotion recognition CLI.",
    no_args_is_help=True,
)
train_app = typer.Typer(help="Stage-1 per-modality encoder training.")
app.add_typer(train_app, name="train")


# ---------------------------------------------------------------------------
# memo train image
# ---------------------------------------------------------------------------


@train_app.command("image")
def train_image(
    data: Path = typer.Option(..., help="Data directory with train.csv (path, label)."),
    epochs: int = typer.Option(15, help="Training epochs."),
    out: Path = typer.Option(Path("checkpoints/image.pt"), help="Checkpoint output path."),
    remap_from: str = typer.Option(
        "fer2013", help="Label remapper: fer2013 | affectnet7 | ekman7."
    ),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
    val_split: float = typer.Option(0.1, help="Validation fraction when val.csv is absent."),
    mixup_alpha: float = typer.Option(
        0.2, help="Mixup α (image-only, last 50% of epochs). Set to 0 to disable."
    ),
) -> None:
    """Train the image (face) encoder on FER2013 / AffectNet-7."""
    from .config import ExperimentConfig
    from .training.train_image import run_train_image

    cfg = ExperimentConfig.from_yaml(config) if config else None
    manifest_path = run_train_image(
        data,
        epochs=epochs,
        out=out,
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        remap_from=remap_from,
        val_split=val_split,
        mixup_alpha=mixup_alpha,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint": str(out)}))


# ---------------------------------------------------------------------------
# memo train text
# ---------------------------------------------------------------------------


@train_app.command("text")
def train_text(
    data: Path = typer.Option(..., help="Data directory with train.csv (text, label)."),
    epochs: int = typer.Option(15, help="Training epochs."),
    out: Path = typer.Option(Path("checkpoints/text.pt"), help="Checkpoint output path."),
    lora: bool = typer.Option(False, help="Enable LoRA r=8 adapters on the last 2 MiniLM layers."),
    remap_from: str = typer.Option("goemotions", help="Label remapper: goemotions | ekman7."),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
    val_split: float = typer.Option(0.1, help="Validation fraction when val.csv is absent."),
) -> None:
    """Train the text encoder on GoEmotions → Ekman-7."""
    from .config import ExperimentConfig
    from .training.train_text import run_train_text

    cfg = ExperimentConfig.from_yaml(config) if config else None
    manifest_path = run_train_text(
        data,
        epochs=epochs,
        out=out,
        lora=lora,
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        remap_from=remap_from,
        val_split=val_split,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint": str(out)}))


# ---------------------------------------------------------------------------
# memo train joint
# ---------------------------------------------------------------------------


@train_app.command("joint")
def train_joint(
    aligned_train: Path = typer.Option(
        ..., "--aligned-train", help="Aligned multimodal train JSONL."
    ),
    aligned_val: Path = typer.Option(..., "--aligned-val", help="Aligned multimodal val JSONL."),
    out: Path = typer.Option(
        Path("checkpoints/joint.pt"), help="Checkpoint stem (per-encoder files)."
    ),
    epochs: int = typer.Option(8, help="Training epochs."),
    aux_lambda: float = typer.Option(0.3, help="Weight on each per-modality auxiliary loss."),
    lora: bool = typer.Option(False, help="Text encoder uses LoRA adapters."),
    remap_from: str = typer.Option("ekman7", help="Label remapper: goemotions | ekman7."),
    image_ckpt: Path | None = typer.Option(None, help="Stage-1 image checkpoint to warm-start."),
    text_ckpt: Path | None = typer.Option(None, help="Stage-1 text checkpoint to warm-start."),
    audio_ckpt: Path | None = typer.Option(None, help="Stage-1 audio checkpoint to warm-start."),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
) -> None:
    """Optionally joint-fine-tune all three encoders on aligned multimodal data."""
    from .config import ExperimentConfig
    from .training.train_joint import run_train_joint

    cfg = ExperimentConfig.from_yaml(config) if config else None
    manifest_path = run_train_joint(
        aligned_train,
        aligned_val,
        out=out,
        epochs=epochs,
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        remap_from=remap_from,
        aux_lambda=aux_lambda,
        lora=lora,
        image_ckpt=image_ckpt,
        text_ckpt=text_ckpt,
        audio_ckpt=audio_ckpt,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint_stem": str(out)}))


# ---------------------------------------------------------------------------
# memo train audio
# ---------------------------------------------------------------------------


@train_app.command("audio")
def train_audio(
    data: Path = typer.Option(..., help="Data directory with train.csv (path, label)."),
    epochs: int = typer.Option(15, help="Training epochs (per fold when --k-fold)."),
    out: Path = typer.Option(Path("checkpoints/audio.pt"), help="Checkpoint output path."),
    k_fold: bool = typer.Option(False, help="Run stratified 5-fold CV (recommended for RAVDESS)."),
    distill: bool = typer.Option(
        False, help="Distill from a frozen Wav2Vec2-Base teacher (§4.4); single split, no k-fold."
    ),
    remap_from: str = typer.Option("ravdess", help="Label remapper: ravdess | cremad | ekman7."),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
    val_split: float = typer.Option(0.1, help="Validation fraction when val.csv is absent."),
) -> None:
    """Train the audio encoder on RAVDESS + CREMA-D (optionally with --distill)."""
    from .config import ExperimentConfig
    from .training.train_audio import run_train_audio

    cfg = ExperimentConfig.from_yaml(config) if config else None
    manifest_path = run_train_audio(
        data,
        epochs=epochs,
        out=out,
        k_fold=k_fold,
        distill=distill,
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        remap_from=remap_from,
        val_split=val_split,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint": str(out)}))
