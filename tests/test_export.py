"""Phase 13 ONNX export tests — FP32 + dynamic-INT8 parity on tiny stub models.

Skips when onnx / onnxruntime aren't installed (they're in the [dev] extra, so
CI's onnx-parity job runs them).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from memo.encoders.base import BaseEncoder  # noqa: E402
from memo.export import (  # noqa: E402
    FP32_TOL,
    INT8_TOL,
    encoder_export_spec,
    export_module,
    export_onnx,
    run_export,
)


class _Stub(nn.Module):
    """Flatten → Linear → 7 logits (enough ops for INT8 weight quantization to bite)."""

    def __init__(self, in_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_AXES = {"x": {0: "batch"}, "logits": {0: "batch"}}


def test_fp32_parity(tmp_path: Path) -> None:
    res = export_module(
        _Stub(), tmp_path / "m.onnx", (torch.randn(1, 8),), input_names=["x"], dynamic_axes=_AXES
    )
    assert res.fp32_path.exists()
    assert res.fp32_mae < FP32_TOL
    assert res.int8_path is None  # quantize defaults off
    assert res.fp32_size_mb > 0.0


def test_int8_parity(tmp_path: Path) -> None:
    res = export_module(
        _Stub(),
        tmp_path / "m.onnx",
        (torch.randn(1, 8),),
        input_names=["x"],
        dynamic_axes=_AXES,
        quantize=True,
    )
    assert res.fp32_mae < FP32_TOL
    assert res.int8_path is not None and res.int8_path.exists()
    assert res.int8_mae is not None and res.int8_mae < INT8_TOL


def test_dynamic_batch_axis(tmp_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    model = _Stub()
    path = tmp_path / "m.onnx"
    export_onnx(model, (torch.randn(1, 8),), path, input_names=["x"], dynamic_axes=_AXES)
    # Exported with batch=1; run with batch=4 to confirm the batch axis is dynamic.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"x": np.random.randn(4, 8).astype(np.float32)})[0]
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 7)


def test_encoder_export_spec() -> None:
    img_inputs, img_names, img_axes = encoder_export_spec("image")
    assert img_inputs[0].shape == (1, 3, 112, 112)
    assert img_names == ["pixel_values"]

    txt_inputs, txt_names, _ = encoder_export_spec("text")
    assert len(txt_inputs) == 2 and txt_names == ["input_ids", "attention_mask"]

    _, _, aud_axes = encoder_export_spec("audio")
    assert aud_axes["logmel"] == {0: "batch", 2: "time"}  # dynamic time axis

    with pytest.raises(ValueError):
        encoder_export_spec("bogus")


# --- to_onnx delegation + run_export integration ---------------------------


class _AudioStub(BaseEncoder):
    """Matches the audio export spec: (B, 64, T) → (B, 7)."""

    name = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=2))  # pool over the (dynamic) time axis


def test_to_onnx_delegates(tmp_path: Path) -> None:
    out = tmp_path / "audio.onnx"
    _AudioStub().to_onnx(out, quantize=False)  # uses encoder_export_spec("audio")
    assert out.exists()


class _ImageStub(BaseEncoder):
    name = "image"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


class _TextStub(BaseEncoder):
    name = "text"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 7)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Use attention_mask so the exporter keeps it as a graph input (the real
        # MiniLM encoder masks too) — otherwise torch.onnx prunes the unused input.
        masked = input_ids.float() * attention_mask.float()
        return self.fc(masked.mean(dim=1, keepdim=True))


def test_run_export_injected(tmp_path: Path) -> None:
    import json

    encoders = {"image": _ImageStub(), "text": _TextStub(), "audio": _AudioStub()}
    summary = run_export(tmp_path / "onnx", encoders=encoders, quantize=True)

    assert summary["quantized"] is True
    assert set(summary["encoders"]) == {"image", "text", "audio"}
    for enc in summary["encoders"].values():
        assert enc["fp32_mae"] < FP32_TOL
        assert enc["int8_mae"] < INT8_TOL
    assert summary["total_size_mb"] > 0.0
    # Report must be valid JSON on disk.
    json.loads((tmp_path / "onnx" / "onnx_export.json").read_text())
