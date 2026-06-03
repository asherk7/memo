"""Per-encoder ONNX export: FP32 + dynamic INT8 with parity checks.

Each encoder exports independently; fusion stays pure-numpy at runtime (7
scalars, no ONNX needed). Parity tolerances vs PyTorch: FP32 MAE < 1e-4, dynamic
INT8 MAE < 5e-2. All encoders export a dynamic batch axis; audio additionally
exports a dynamic time axis (variable-length log-mel).

`export_module` works on any `nn.Module`; `export_encoders` wires the three real
encoders with their per-modality input specs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from loguru import logger
from torch import nn

__all__ = [
    "ExportResult",
    "OnnxParityError",
    "export_onnx",
    "quantize_dynamic_int8",
    "onnx_parity_mae",
    "export_module",
    "encoder_export_spec",
    "export_encoders",
    "run_export",
    "FP32_TOL",
    "INT8_TOL",
]

FP32_TOL = 1e-4
INT8_TOL = 5e-2
_OPSET = 17


class OnnxParityError(RuntimeError):
    """An exported ONNX graph's output diverged from PyTorch beyond tolerance."""


@dataclass(frozen=True)
class ExportResult:
    """Outcome of exporting one module: paths, parity MAEs, and on-disk sizes."""

    name: str
    fp32_path: Path
    fp32_mae: float
    fp32_size_mb: float
    int8_path: Path | None = None
    int8_mae: float | None = None
    int8_size_mb: float | None = None


def _to_numpy(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def export_onnx(
    model: nn.Module,
    example_inputs: Sequence[Any],
    path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str] = ("logits",),
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset: int = _OPSET,
) -> None:
    """Export ``model.forward(*example_inputs)`` to an ONNX graph at ``path``."""
    import warnings

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), warnings.catch_warnings():
        # Use the stable TorchScript exporter (dynamo=False); the dynamo exporter
        # is still flaky on transformers models. torch 2.9 flipped the default to
        # dynamo=True and deprecates this path. Scope the deprecation filter to
        # torch.onnx so a model's own DeprecationWarnings still surface.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx.*")
        torch.onnx.export(
            model,
            tuple(example_inputs),
            str(path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes={k: dict(v) for k, v in dynamic_axes.items()} if dynamic_axes else None,
            opset_version=opset,
            dynamo=False,
        )


def quantize_dynamic_int8(fp32_path: Path, int8_path: Path) -> None:
    """Dynamic INT8 quantization (per-channel weights) via onnxruntime."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def onnx_parity_mae(
    model: nn.Module,
    onnx_path: Path,
    example_inputs: Sequence[Any],
    input_names: Sequence[str],
) -> float:
    """Mean-absolute-error between the PyTorch output and the ONNX-runtime output."""
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        reference = _to_numpy(model(*example_inputs))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Feed only inputs the graph actually declares: the exporter prunes inputs a
    # model never reads, so a stale name would otherwise make ORT raise.
    declared = {i.name for i in session.get_inputs()}
    feed = {
        name: _to_numpy(t)
        for name, t in zip(input_names, example_inputs, strict=True)
        if name in declared
    }
    output = session.run(None, feed)[0]
    return float(np.abs(reference - output).mean())


def export_module(
    model: nn.Module,
    fp32_path: Path,
    example_inputs: Sequence[Any],
    *,
    input_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    quantize: bool = False,
    name: str = "module",
    fp32_tol: float = FP32_TOL,
    int8_tol: float = INT8_TOL,
) -> ExportResult:
    """Export FP32 (+ optional INT8), parity-check each, and enforce tolerances.

    Leaves ``model`` in eval mode (export requires it); callers mid-experiment
    should restore training mode. Raises `OnnxParityError` if a graph's output
    diverges from PyTorch beyond ``fp32_tol`` / ``int8_tol``.
    """
    fp32_path = Path(fp32_path)
    export_onnx(
        model, example_inputs, fp32_path, input_names=input_names, dynamic_axes=dynamic_axes
    )
    fp32_mae = onnx_parity_mae(model, fp32_path, example_inputs, input_names)
    if fp32_mae >= fp32_tol:
        raise OnnxParityError(f"{name}: FP32 ONNX parity MAE {fp32_mae:.2e} ≥ tol {fp32_tol:.0e}")

    result = ExportResult(
        name=name,
        fp32_path=fp32_path,
        fp32_mae=fp32_mae,
        fp32_size_mb=_size_mb(fp32_path),
    )
    if not quantize:
        return result

    int8_path = fp32_path.with_name(f"{fp32_path.stem}.int8.onnx")
    quantize_dynamic_int8(fp32_path, int8_path)
    int8_mae = onnx_parity_mae(model, int8_path, example_inputs, input_names)
    if int8_mae >= int8_tol:
        raise OnnxParityError(f"{name}: INT8 ONNX parity MAE {int8_mae:.2e} ≥ tol {int8_tol:.0e}")

    return replace(
        result,
        int8_path=int8_path,
        int8_mae=int8_mae,
        int8_size_mb=_size_mb(int8_path),
    )


def encoder_export_spec(
    name: str,
) -> tuple[tuple[Any, ...], list[str], dict[str, dict[int, str]]]:
    """Per-modality (example_inputs, input_names, dynamic_axes) for ONNX export.

    Tensors are freshly built (no import-time randomness). Image/audio take a bare
    tensor; text takes (input_ids, attention_mask). Audio's time axis is dynamic.
    """
    if name == "image":
        return (
            (torch.randn(1, 3, 112, 112),),
            ["pixel_values"],
            {"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        )
    if name == "text":
        return (
            (torch.randint(0, 1000, (1, 16)), torch.ones(1, 16, dtype=torch.long)),
            ["input_ids", "attention_mask"],
            {
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )
    if name == "audio":
        return (
            (torch.randn(1, 64, 301),),
            ["logmel"],
            {"logmel": {0: "batch", 2: "time"}, "logits": {0: "batch"}},
        )
    raise ValueError(f"No export spec for encoder {name!r}; expected image | text | audio.")


def export_encoders(
    encoders: Mapping[str, nn.Module],
    out_dir: Path,
    *,
    quantize: bool = False,
) -> dict[str, ExportResult]:
    """Export each named encoder to ``out_dir/<name>.onnx`` (+ INT8 when requested)."""
    out_dir = Path(out_dir)
    results: dict[str, ExportResult] = {}
    for name, model in encoders.items():
        example_inputs, input_names, dynamic_axes = encoder_export_spec(name)
        results[name] = export_module(
            model,
            out_dir / f"{name}.onnx",
            example_inputs,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
            quantize=quantize,
            name=name,
        )
        r = results[name]
        logger.info(
            "{}: FP32 {:.1f} MB (MAE {:.1e}){}",
            name,
            r.fp32_size_mb,
            r.fp32_mae,
            f" · INT8 {r.int8_size_mb:.1f} MB (MAE {r.int8_mae:.1e})" if r.int8_path else "",
        )
    return results


def run_export(
    out_dir: Path,
    *,
    config_path: Path | None = None,
    encoders: Mapping[str, nn.Module] | None = None,
    quantize: bool = False,
) -> dict[str, Any]:
    """Export the three encoders to ONNX; write ``onnx_export.json``; return a summary.

    Either inject ``encoders`` (tests) or pass ``config_path`` to load the real
    three via `MultimodalEmotionPipeline.from_config`.
    """
    if encoders is None:
        if config_path is None:
            raise ValueError("run_export needs either config_path or encoders.")
        from .pipeline import MultimodalEmotionPipeline

        pipe = MultimodalEmotionPipeline.from_config(config_path)
        # The real encoders are nn.Module subclasses (typed as the ModalityEncoder
        # Protocol on the pipeline); cast so the export typing lines up.
        encoders = cast(
            "Mapping[str, nn.Module]",
            {"image": pipe.image_encoder, "text": pipe.text_encoder, "audio": pipe.audio_encoder},
        )

    out_dir = Path(out_dir)
    results = export_encoders(encoders, out_dir, quantize=quantize)

    summary: dict[str, Any] = {
        "quantized": quantize,
        "encoders": {
            name: {
                "fp32_path": str(r.fp32_path),
                "fp32_mae": r.fp32_mae,
                "fp32_size_mb": r.fp32_size_mb,
                "int8_path": str(r.int8_path) if r.int8_path else None,
                "int8_mae": r.int8_mae,
                "int8_size_mb": r.int8_size_mb,
            }
            for name, r in results.items()
        },
    }
    size_key = "int8_size_mb" if quantize else "fp32_size_mb"
    summary["total_size_mb"] = sum(getattr(r, size_key) or 0.0 for r in results.values())

    (out_dir / "onnx_export.json").write_text(json.dumps(summary, indent=2))
    return summary
