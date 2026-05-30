"""Phase 1 acceptance smoke tests.

These codify the criteria listed in ROADMAP.md Phase 1 directly so a green
`pytest -q` confirms the scaffold is wired correctly.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from memo import EmotionPrediction
from memo.config import ExperimentConfig
from memo.labels import (
    NUM_CLASSES,
    EkmanEmotion,
    remap_cremad,
    remap_fer2013,
    remap_goemotions,
    remap_ravdess,
)
from memo.seed import seed_everything


def test_ekman_label_values() -> None:
    assert EkmanEmotion.ANGER == 0
    assert EkmanEmotion.DISGUST == 1
    assert EkmanEmotion.FEAR == 2
    assert EkmanEmotion.HAPPINESS == 3
    assert EkmanEmotion.SADNESS == 4
    assert EkmanEmotion.SURPRISE == 5
    assert EkmanEmotion.NEUTRAL == 6
    assert NUM_CLASSES == 7
    assert len(list(EkmanEmotion)) == NUM_CLASSES


def test_dataset_remappers() -> None:
    # FER2013 is identity by construction.
    assert remap_fer2013(3) is EkmanEmotion.HAPPINESS
    # GoEmotions: by name and by index.
    assert remap_goemotions("joy") is EkmanEmotion.HAPPINESS
    assert remap_goemotions("disgust") is EkmanEmotion.DISGUST
    assert remap_goemotions(27) is EkmanEmotion.NEUTRAL  # "neutral" is the last entry
    # RAVDESS — calm collapses to neutral.
    assert remap_ravdess(2) is EkmanEmotion.NEUTRAL
    assert remap_ravdess(5) is EkmanEmotion.ANGER
    # CREMA-D.
    assert remap_cremad("ANG") is EkmanEmotion.ANGER
    assert remap_cremad("hap") is EkmanEmotion.HAPPINESS  # case-insensitive


def test_emotion_prediction_is_frozen() -> None:
    pred = EmotionPrediction(
        label=EkmanEmotion.HAPPINESS,
        probs={e: 0.0 for e in EkmanEmotion},
        per_modality_probs={"image": {e: 0.0 for e in EkmanEmotion}},
        confidences={"image": 0.9},
        gate_weights={"image": 1.0},
        used_modalities=("image",),
        abstained=False,
    )
    assert pred.label is EkmanEmotion.HAPPINESS
    with pytest.raises(dataclasses.FrozenInstanceError):
        pred.abstained = True  # type: ignore[misc]


def test_seed_torch_reproducible_in_one_process() -> None:
    seed_everything(42)
    a = torch.randn(3)
    seed_everything(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_seed_numpy_reproducible_in_one_process() -> None:
    seed_everything(42)
    a = np.random.rand(3)
    seed_everything(42)
    b = np.random.rand(3)
    assert np.array_equal(a, b)


def test_seed_reproducible_across_interpreters(tmp_path: Path) -> None:
    """Phase 1 acceptance: two fresh interpreters must produce identical draws."""
    script = tmp_path / "draw.py"
    script.write_text(
        "import numpy as np, torch\n"
        "from memo.seed import seed_everything\n"
        "seed_everything(42)\n"
        "print(','.join(f'{x:.10f}' for x in torch.randn(3).tolist()))\n"
        "print(','.join(f'{x:.10f}' for x in np.random.rand(3).tolist()))\n"
    )
    out1 = subprocess.check_output([sys.executable, str(script)], text=True)
    out2 = subprocess.check_output([sys.executable, str(script)], text=True)
    assert out1 == out2


def test_experiment_config_loads_default_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ExperimentConfig.from_yaml(root / "configs" / "default.yaml")
    # Spot-check structural integrity rather than every field.
    assert cfg.seed == 42
    assert cfg.model.encoders.image.image_size == 112
    assert cfg.model.encoders.text.backbone == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.model.fusion.abstention_threshold == 0.40
    assert cfg.model.kd.enabled is False
    assert cfg.train.optimizer.backbone_lr == 1.0e-5
    assert cfg.train.modality_dropout.text_rate == 0.15


def test_experiment_config_defaults_when_yaml_partial(tmp_path: Path) -> None:
    """Missing keys must fall back to the dataclass default — not crash."""
    p = tmp_path / "partial.yaml"
    p.write_text("seed: 7\n")
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.seed == 7
    assert cfg.model.fusion.abstention_threshold == 0.40  # default kept


def test_experiment_config_drops_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    p.write_text("seed: 1\nbogus_top_level_key: 99\n")
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.seed == 1


def test_public_api_imports() -> None:
    """Acceptance criterion: the documented imports must succeed."""
    import memo
    from memo import EmotionPrediction  # noqa: F401
    from memo.labels import EkmanEmotion

    assert EkmanEmotion.HAPPINESS == 3
    assert hasattr(memo, "__version__")
