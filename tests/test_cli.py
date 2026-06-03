"""CLI tests — `memo predict` JSON output, arg validation, help surface."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import memo.pipeline as pipeline_mod
from memo.cli import _prediction_to_json, app
from memo.labels import EkmanEmotion
from memo.types import EmotionPrediction

runner = CliRunner()


def _sample_prediction() -> EmotionPrediction:
    probs = {e: (1.0 if e is EkmanEmotion.HAPPINESS else 0.0) for e in EkmanEmotion}
    return EmotionPrediction(
        label=EkmanEmotion.HAPPINESS,
        probs=probs,
        per_modality_probs={"text": probs},
        confidences={"text": 0.9},
        gate_weights={"text": 1.0},
        used_modalities=("text",),
        abstained=False,
    )


def test_prediction_to_json_is_serializable() -> None:
    d = _prediction_to_json(_sample_prediction())
    json.dumps(d)  # must not raise (no enums / tensors leaking)
    assert d["label"] == "happiness"
    assert set(d["probs"]) == {e.name.lower() for e in EkmanEmotion}
    assert d["used_modalities"] == ["text"]
    assert d["abstained"] is False


def test_predict_requires_an_input() -> None:
    result = runner.invoke(app, ["predict"])
    assert result.exit_code != 0  # BadParameter: needs ≥1 modality


class _StubPipe:
    def predict(self, **_kwargs: object) -> EmotionPrediction:
        return _sample_prediction()


def _stub_from_config(monkeypatch: pytest.MonkeyPatch, pipe: object) -> None:
    monkeypatch.setattr(
        pipeline_mod.MultimodalEmotionPipeline,
        "from_config",
        classmethod(lambda cls, path: pipe),
    )


def test_predict_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_from_config(monkeypatch, _StubPipe())
    result = runner.invoke(app, ["predict", "--text", "I am thrilled"])
    assert result.exit_code == 0, result.output
    # The untrained-checkpoint warning goes to stderr; parse the JSON object itself.
    out = json.loads(result.output[result.output.index("{") :])
    assert out["label"] == "happiness"
    assert out["used_modalities"] == ["text"]


def test_predict_missing_audio_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _stub_from_config(monkeypatch, _StubPipe())
    result = runner.invoke(app, ["predict", "--audio", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0  # clean BadParameter, not a soundfile traceback
    assert "could not read audio" in result.output


def test_predict_pipeline_valueerror_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RaisingPipe:
        def predict(self, **_kwargs: object) -> EmotionPrediction:
            raise ValueError("no usable modality")

    _stub_from_config(monkeypatch, _RaisingPipe())
    result = runner.invoke(app, ["predict", "--text", "x"])
    assert result.exit_code != 0
    assert "no usable modality" in result.output  # surfaced as BadParameter, not a traceback


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("predict", "train", "calibrate", "evaluate", "benchmark", "export"):
        assert command in result.output
