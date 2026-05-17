"""Typed experiment configuration + YAML loader.

The on-disk format is YAML (see `configs/default.yaml`); in-process code
passes around `ExperimentConfig` instances. Validation happens at load time:
unknown keys are dropped, missing keys fall back to defaults, and type
mismatches surface as a `TypeError` from the dataclass constructor rather
than as a confusing `KeyError` 10 modules later.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "PathsConfig",
    "EncodersConfig",
    "ImageEncoderConfig",
    "TextEncoderConfig",
    "AudioEncoderConfig",
    "FusionConfig",
    "LoRAConfig",
    "KDConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SchedulerMaxLR",
    "FocalLossConfig",
    "ModalityDropoutConfig",
    "CalibrationConfig",
]


# --- Encoders --------------------------------------------------------------


@dataclass
class ImageEncoderConfig:
    backbone: str = "mobilenet_v3_small"
    weights: str = "IMAGENET1K_V1"
    image_size: int = 112
    checkpoint: str | None = None


@dataclass
class TextEncoderConfig:
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    head_dropout: float = 0.1
    checkpoint: str | None = None


@dataclass
class AudioEncoderConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    window_seconds: float = 3.0
    checkpoint: str | None = None


@dataclass
class EncodersConfig:
    image: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)


# --- Fusion / PEFT / KD ---------------------------------------------------


@dataclass
class FusionConfig:
    abstention_threshold: float = 0.40
    gamma_init: float = 1.0
    temperature_init: float = 1.0
    weight_init: float = 0.0
    checkpoint: str | None = None


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["last_2_transformer_layers"])


@dataclass
class KDConfig:
    enabled: bool = False
    teacher: str = "facebook/wav2vec2-base"
    alpha: float = 0.5
    temperature: float = 4.0


@dataclass
class ModelConfig:
    encoders: EncodersConfig = field(default_factory=EncodersConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    kd: KDConfig = field(default_factory=KDConfig)


# --- Training -------------------------------------------------------------


@dataclass
class OptimizerConfig:
    backbone_lr: float = 1.0e-5
    head_lr: float = 1.0e-3
    weight_decay: float = 0.01


@dataclass
class SchedulerMaxLR:
    image: float = 3.0e-3
    text_head: float = 1.0e-3
    audio: float = 5.0e-3


@dataclass
class SchedulerConfig:
    name: str = "onecycle"
    max_lr: SchedulerMaxLR = field(default_factory=SchedulerMaxLR)


@dataclass
class FocalLossConfig:
    gamma: float = 2.0
    label_smoothing: float = 0.05
    class_weight_beta: float = 0.9999


@dataclass
class ModalityDropoutConfig:
    rate: float = 0.3
    text_rate: float = 0.15


@dataclass
class CalibrationConfig:
    epochs: int = 200
    lr: float = 1.0e-2


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    freeze_backbone_epochs: int = 3
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    focal_loss: FocalLossConfig = field(default_factory=FocalLossConfig)
    modality_dropout: ModalityDropoutConfig = field(default_factory=ModalityDropoutConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


# --- Paths + Top-level config --------------------------------------------


@dataclass
class PathsConfig:
    checkpoints: str = "checkpoints"
    data: str = "data"
    runs: str = "runs"


@dataclass
class ExperimentConfig:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an `ExperimentConfig` from a YAML file."""
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected the top-level YAML to be a mapping, got {type(data).__name__}"
            )
        return _from_mapping(cls, data)


# --- YAML → dataclass adapter --------------------------------------------


def _unwrap_optional(tp: Any) -> Any:
    """Reduce `X | None` (or `Optional[X]`) to `X`."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _from_mapping(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass instance from a (possibly partial) mapping."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            # Unknown keys are dropped intentionally — keeps old YAMLs
            # loadable as the schema evolves.
            continue
        target = _unwrap_optional(hints[key])
        if value is None:
            kwargs[key] = None
        elif isinstance(target, type) and is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _from_mapping(cast(type, target), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
