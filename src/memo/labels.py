"""Ekman-7 emotion labels and dataset → Ekman remappers.

The `EkmanEmotion` integer ordering here is load-bearing: every loss weight,
confusion matrix, head output, and dataset remapper in the rest of the codebase
indexes against this exact order. Do not reorder.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "EkmanEmotion",
    "NUM_CLASSES",
    "EMOTION_NAMES",
    "remap_fer2013",
    "remap_goemotions",
    "remap_ravdess",
    "remap_cremad",
]


class EkmanEmotion(IntEnum):
    """The 7 Ekman emotion labels in their canonical training order."""

    ANGER = 0
    DISGUST = 1
    FEAR = 2
    HAPPINESS = 3
    SADNESS = 4
    SURPRISE = 5
    NEUTRAL = 6


NUM_CLASSES: int = 7

EMOTION_NAMES: tuple[str, ...] = tuple(e.name.lower() for e in EkmanEmotion)


# --- Dataset remappers ---------------------------------------------------
# Each remapper takes the dataset-native integer (or string) label and returns
# the `EkmanEmotion`. Out-of-distribution labels raise `KeyError` so silent
# data corruption can't hide in a training pipeline.


_FER2013: dict[int, EkmanEmotion] = {
    0: EkmanEmotion.ANGER,
    1: EkmanEmotion.DISGUST,
    2: EkmanEmotion.FEAR,
    3: EkmanEmotion.HAPPINESS,
    4: EkmanEmotion.SADNESS,
    5: EkmanEmotion.SURPRISE,
    6: EkmanEmotion.NEUTRAL,
}


def remap_fer2013(label: int) -> EkmanEmotion:
    """FER2013 ships labels in the same Ekman-7 order — identity map, kept
    explicit so a reorder of `EkmanEmotion` would surface here."""
    return _FER2013[int(label)]


# GoEmotions: 28 fine-grained → Ekman-7 via the standard categorical mapping.
# Indices follow `huggingface/datasets goemotions` simple config.
_GOEMOTIONS_NAME_TO_EKMAN: dict[str, EkmanEmotion] = {
    "admiration": EkmanEmotion.HAPPINESS,
    "amusement": EkmanEmotion.HAPPINESS,
    "approval": EkmanEmotion.HAPPINESS,
    "caring": EkmanEmotion.HAPPINESS,
    "desire": EkmanEmotion.HAPPINESS,
    "excitement": EkmanEmotion.HAPPINESS,
    "gratitude": EkmanEmotion.HAPPINESS,
    "joy": EkmanEmotion.HAPPINESS,
    "love": EkmanEmotion.HAPPINESS,
    "optimism": EkmanEmotion.HAPPINESS,
    "pride": EkmanEmotion.HAPPINESS,
    "relief": EkmanEmotion.HAPPINESS,
    "anger": EkmanEmotion.ANGER,
    "annoyance": EkmanEmotion.ANGER,
    "disapproval": EkmanEmotion.ANGER,
    "disgust": EkmanEmotion.DISGUST,
    "fear": EkmanEmotion.FEAR,
    "nervousness": EkmanEmotion.FEAR,
    "sadness": EkmanEmotion.SADNESS,
    "disappointment": EkmanEmotion.SADNESS,
    "embarrassment": EkmanEmotion.SADNESS,
    "grief": EkmanEmotion.SADNESS,
    "remorse": EkmanEmotion.SADNESS,
    "surprise": EkmanEmotion.SURPRISE,
    "confusion": EkmanEmotion.SURPRISE,
    "curiosity": EkmanEmotion.SURPRISE,
    "realization": EkmanEmotion.SURPRISE,
    "neutral": EkmanEmotion.NEUTRAL,
}

_GOEMOTIONS_ORDER: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


def remap_goemotions(label: int | str) -> EkmanEmotion:
    """Accepts the integer label (HF datasets order) or the string name."""
    if isinstance(label, str):
        return _GOEMOTIONS_NAME_TO_EKMAN[label]
    return _GOEMOTIONS_NAME_TO_EKMAN[_GOEMOTIONS_ORDER[int(label)]]


# RAVDESS filename code (the 3rd field of "Modality-Voice-Emotion-..."):
#   01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
# We collapse `calm` into `neutral` (no Ekman class for it).
_RAVDESS: dict[int, EkmanEmotion] = {
    1: EkmanEmotion.NEUTRAL,
    2: EkmanEmotion.NEUTRAL,
    3: EkmanEmotion.HAPPINESS,
    4: EkmanEmotion.SADNESS,
    5: EkmanEmotion.ANGER,
    6: EkmanEmotion.FEAR,
    7: EkmanEmotion.DISGUST,
    8: EkmanEmotion.SURPRISE,
}


def remap_ravdess(code: int) -> EkmanEmotion:
    return _RAVDESS[int(code)]


# CREMA-D 3-letter codes (no surprise class).
_CREMAD: dict[str, EkmanEmotion] = {
    "ANG": EkmanEmotion.ANGER,
    "DIS": EkmanEmotion.DISGUST,
    "FEA": EkmanEmotion.FEAR,
    "HAP": EkmanEmotion.HAPPINESS,
    "NEU": EkmanEmotion.NEUTRAL,
    "SAD": EkmanEmotion.SADNESS,
}


def remap_cremad(code: str) -> EkmanEmotion:
    return _CREMAD[code.upper()]
