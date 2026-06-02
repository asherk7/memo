"""MEMO command-line interface (§8).

Entry point registered as ``memo`` in ``pyproject.toml``.  Subcommands are
wired per-phase; later phases add their own commands to ``train_app`` and the
top-level ``app`` directly.

Currently wired:
    memo train image  --data <dir> --epochs N --out <ckpt>
    memo train text   --data <dir> --epochs N --out <ckpt>
    memo train audio  --data <dir> --epochs N [--distill] --out <ckpt>
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
    remap_from: str = typer.Option("fer2013", help="Label remapper: fer2013 | ekman7."),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
    val_split: float = typer.Option(0.1, help="Validation fraction when val.csv is absent."),
) -> None:
    """Train the image (face) encoder on FER2013."""
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
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        remap_from=remap_from,
        val_split=val_split,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint": str(out)}))


# ---------------------------------------------------------------------------
# memo train audio
# ---------------------------------------------------------------------------


@train_app.command("audio")
def train_audio(
    data: Path = typer.Option(..., help="Data directory with train.csv (path, label)."),
    epochs: int = typer.Option(15, help="Training epochs."),
    out: Path = typer.Option(Path("checkpoints/audio.pt"), help="Checkpoint output path."),
    distill: bool = typer.Option(
        False, help="Distill from a frozen Wav2Vec2-Base teacher (§4.4)."
    ),
    remap_from: str = typer.Option("ravdess", help="Label remapper: ravdess | cremad | ekman7."),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
    val_split: float = typer.Option(0.1, help="Validation fraction when val.csv is absent."),
) -> None:
    """Train the audio encoder on RAVDESS (optionally with --distill)."""
    from .config import ExperimentConfig
    from .training.train_audio import run_train_audio

    cfg = ExperimentConfig.from_yaml(config) if config else None
    manifest_path = run_train_audio(
        data,
        epochs=epochs,
        out=out,
        distill=distill,
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        remap_from=remap_from,
        val_split=val_split,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint": str(out)}))


# ---------------------------------------------------------------------------
# memo calibrate  (Stage 3 / Phase 11 — fusion calibration)
# ---------------------------------------------------------------------------


@app.command("calibrate")
def calibrate(
    aligned_val: Path = typer.Option(..., "--aligned-val", help="Aligned multimodal val JSONL."),
    out: Path = typer.Option(
        Path("checkpoints/fusion.pt"), help="Calibrated LateFusion checkpoint output."
    ),
    modality_dropout: float = typer.Option(
        0.3, help="Per-sample modality-dropout rate (text uses its own asymmetric rate)."
    ),
    remap_from: str = typer.Option("ekman7", help="Aligned-label remapper: ekman7 | ravdess."),
    config: Path | None = typer.Option(None, help="Optional YAML config path."),
    device: str = typer.Option("cpu", help="Torch device."),
    runs_dir: Path = typer.Option(Path("runs"), help="Root dir for run artifacts."),
) -> None:
    """Calibrate the 7 fusion scalars under modality dropout (Stage 3)."""
    from .config import ExperimentConfig
    from .training.calibrate_fusion import run_calibrate_fusion

    cfg = ExperimentConfig.from_yaml(config) if config else None
    manifest_path = run_calibrate_fusion(
        aligned_val,
        out=out,
        config=cfg,
        device=device,
        runs_dir=runs_dir,
        modality_dropout=modality_dropout,
        remap_from=remap_from,
    )
    typer.echo(json.dumps({"manifest": str(manifest_path), "checkpoint": str(out)}))


# ---------------------------------------------------------------------------
# memo evaluate / benchmark  (Phase 12 — measurement layer)
# ---------------------------------------------------------------------------


@app.command("evaluate")
def evaluate(
    aligned_test: Path = typer.Option(..., "--aligned-test", help="Aligned multimodal test JSONL."),
    config: Path = typer.Option(..., help="YAML config pointing at the calibrated checkpoints."),
    out: Path = typer.Option(Path("runs/eval"), help="Output dir for report.{json,md}."),
    remap_from: str = typer.Option("ekman7", help="Aligned-label remapper: ekman7 | ravdess."),
    device: str = typer.Option("cpu", help="Torch device."),
) -> None:
    """Headline metrics + 7-subset ablation + gating + robustness sweep."""
    from .eval.evaluate import run_evaluate

    report = run_evaluate(
        aligned_test, out_dir=out, config_path=config, remap_from=remap_from, device=device
    )
    typer.echo(json.dumps({"out": str(out), "macro_f1": report["headline"]["macro_f1"]}))


@app.command("benchmark")
def benchmark(
    config: Path = typer.Option(..., help="YAML config (loads the encoders to benchmark)."),
    runs: int = typer.Option(100, help="Timed runs per encoder."),
    out: Path = typer.Option(Path("runs/bench.json"), help="Output JSON path."),
) -> None:
    """Per-encoder CPU latency / params / MACs + peak RSS."""
    from .eval.benchmark import run_benchmark

    results = run_benchmark(config_path=config, runs=runs, out=out)
    typer.echo(json.dumps({"out": str(out), "peak_rss_mb": results["peak_rss_mb"]}))
