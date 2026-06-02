"""MEMO command-line interface (§8).

Entry point registered as ``memo`` in ``pyproject.toml``. Every subcommand prints
a single JSON object to stdout so it composes with shell pipelines.

Full surface:
    memo predict   [--image f] [--text s] [--audio f] [--config y]   → prediction JSON
    memo train image  --data <dir> --epochs N --out <ckpt>
    memo train text   --data <dir> --epochs N --out <ckpt>
    memo train audio  --data <dir> --epochs N [--distill] --out <ckpt>
    memo calibrate --aligned-val v.jsonl --out <ckpt>                 (Stage 3)
    memo evaluate  --aligned-test t.jsonl --config y --out <dir>
    memo benchmark --config y --runs N --out <json>
    memo export    --config y --out <dir> [--quantize int8]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="memo",
    help="MEMO multimodal emotion recognition CLI.",
    no_args_is_help=True,
)
train_app = typer.Typer(help="Stage-1 per-modality encoder training.")
app.add_typer(train_app, name="train")


# ---------------------------------------------------------------------------
# memo predict
# ---------------------------------------------------------------------------


def _prediction_to_json(pred: Any) -> dict[str, Any]:
    """Serialize an `EmotionPrediction` to a JSON-friendly dict (enum → lower name)."""

    def _probs(d: dict[Any, float]) -> dict[str, float]:
        return {emotion.name.lower(): value for emotion, value in d.items()}

    return {
        "label": pred.label.name.lower(),
        "probs": _probs(pred.probs),
        "per_modality_probs": {m: _probs(p) for m, p in pred.per_modality_probs.items()},
        "confidences": {m: float(v) for m, v in pred.confidences.items()},
        "gate_weights": {m: float(v) for m, v in pred.gate_weights.items()},
        "used_modalities": list(pred.used_modalities),
        "abstained": pred.abstained,
    }


@app.command("predict")
def predict(
    image: Path | None = typer.Option(None, help="Face image file."),
    text: str | None = typer.Option(None, help="Utterance text."),
    audio: Path | None = typer.Option(None, help="Speech audio (WAV) file."),
    config: Path = typer.Option(
        Path("configs/default.yaml"), help="YAML config (encoder + calibrated checkpoints)."
    ),
) -> None:
    """Predict emotion from any non-empty subset of image / text / audio (JSON to stdout)."""
    if image is None and text is None and audio is None:
        raise typer.BadParameter("supply at least one of --image, --text, --audio.")

    from .config import ExperimentConfig
    from .pipeline import MultimodalEmotionPipeline
    from .preprocessing.audio import SAMPLE_RATE

    enc = ExperimentConfig.from_yaml(config).model
    if all(
        c is None
        for c in (
            enc.encoders.image.checkpoint,
            enc.encoders.text.checkpoint,
            enc.encoders.audio.checkpoint,
            enc.fusion.checkpoint,
        )
    ):
        typer.echo("warning: all checkpoints are null — output is from untrained heads.", err=True)

    pipe = MultimodalEmotionPipeline.from_config(config)
    image_arr = _read_image(image) if image is not None else None
    if audio is not None:
        audio_arr, sample_rate = _read_audio(audio)
    else:
        audio_arr, sample_rate = None, SAMPLE_RATE

    try:
        pred = pipe.predict(
            image=image_arr, text=text, audio=audio_arr, audio_sample_rate=sample_rate
        )
    except ValueError as exc:
        # e.g. an image-only call where no face was detected — degrade to a clear
        # CLI error instead of a raw traceback.
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(_prediction_to_json(pred), indent=2))


def _read_image(path: Path) -> Any:
    """Read an image file → (H, W, 3) RGB array (pipeline does the face crop)."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise typer.BadParameter(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_audio(path: Path) -> tuple[Any, int]:
    """Read an audio file → (mono float32 waveform, sample_rate)."""
    import soundfile as sf

    try:
        waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # missing / unreadable file → clean CLI error, not a traceback
        raise typer.BadParameter(f"could not read audio: {path}") from exc
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, int(sample_rate)


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
    distill: bool = typer.Option(False, help="Distill from a frozen Wav2Vec2-Base teacher (§4.4)."),
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


# ---------------------------------------------------------------------------
# memo export  (Phase 13 — ONNX FP32 + INT8)
# ---------------------------------------------------------------------------


@app.command("export")
def export(
    config: Path = typer.Option(..., help="YAML config (loads the encoders to export)."),
    out: Path = typer.Option(Path("onnx"), help="Output directory for ONNX files."),
    quantize: str = typer.Option("none", help="Quantization: none | int8."),
) -> None:
    """Export each encoder to ONNX FP32 (and optional INT8) with parity checks."""
    from .export import run_export

    summary = run_export(out, config_path=config, quantize=quantize == "int8")
    typer.echo(json.dumps({"out": str(out), "total_size_mb": summary["total_size_mb"]}))
