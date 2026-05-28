"""Dataset adapters: single-modality CSV and aligned multimodal JSONL (§8).

Both adapters are deliberately thin: they parse the manifest file, remap each
dataset-native label to `EkmanEmotion`, and defer the actual modality loading
to an injected ``loader`` callable (so the dataset stays decoupled from the
preprocessing specifics each modality needs). Both expose ``.labels`` as an
int list so `ClassBalancedSampler` and the stratified k-fold runner can read the
class distribution without materializing every sample.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset, Subset

from ..labels import EkmanEmotion

if TYPE_CHECKING:
    from ..config import ExperimentConfig
    from ..losses import FocalLoss

__all__ = [
    "CsvDataset",
    "JsonlDataset",
    "MODALITY_KEYS",
    "stratified_train_val_split",
    "focal_loss_from_labels",
]

MODALITY_KEYS: tuple[str, ...] = ("image", "text", "audio")

# A loader turns a raw cell (a file path or a text string) into an encoder-ready
# input; a remapper maps a dataset-native label to its EkmanEmotion.
Loader = Callable[[Any], Any]
Remap = Callable[[Any], EkmanEmotion]


def _resolve(root: Path | None, value: str) -> str:
    """Join a relative path against ``root`` when one is given."""
    if root is None:
        return value
    p = Path(value)
    return str(p if p.is_absolute() else root / p)


class CsvDataset(Dataset):
    """Single-modality CSV adapter: one row per example.

    The CSV must have a value column (a file path or text) and a label column.
    ``loader`` turns the value into the encoder input; ``remap`` maps the native
    label to `EkmanEmotion`. ``root`` is prepended to relative path values.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        loader: Loader,
        remap: Remap | None = None,
        value_column: str = "path",
        label_column: str = "label",
        root: str | Path | None = None,
        is_path: bool = True,
    ) -> None:
        self.loader = loader
        self.is_path = is_path
        self.root = Path(root) if root is not None else None

        self._values: list[str] = []
        self.labels: list[int] = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or value_column not in reader.fieldnames:
                raise ValueError(f"CSV {csv_path} is missing a '{value_column}' column.")
            if label_column not in reader.fieldnames:
                raise ValueError(f"CSV {csv_path} is missing a '{label_column}' column.")
            for row in reader:
                self._values.append(row[value_column])
                native = row[label_column]
                label = remap(native) if remap is not None else EkmanEmotion(int(native))
                self.labels.append(int(label))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        value = self._values[idx]
        if self.is_path:
            value = _resolve(self.root, value)
        return self.loader(value), self.labels[idx]


class JsonlDataset(Dataset):
    """Aligned multimodal JSONL adapter (§8 record schema).

    One JSON object per line: ``id``, optional ``image`` / ``text`` / ``audio``,
    ``label``, and an optional ``slices`` dict (used by the fairness audit in
    Phase 12). ``loaders`` maps each present modality to its loader callable;
    a record missing a modality yields ``None`` for it (so the pipeline drops it).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        loaders: dict[str, Loader],
        remap: Remap | None = None,
        root: str | Path | None = None,
    ) -> None:
        unknown = set(loaders) - set(MODALITY_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown modality loaders {sorted(unknown)}; expected {MODALITY_KEYS}."
            )
        self.loaders = loaders
        self.root = Path(root) if root is not None else None

        self._records: list[dict[str, Any]] = []
        self.labels: list[int] = []
        self.slices: list[dict[str, Any]] = []
        with open(jsonl_path) as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "label" not in rec:
                    raise ValueError(f"{jsonl_path}:{line_no} record is missing 'label'.")
                native = rec["label"]
                label = remap(native) if remap is not None else EkmanEmotion(int(native))
                self._records.append(rec)
                self.labels.append(int(label))
                self.slices.append(rec.get("slices", {}))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], int]:
        rec = self._records[idx]
        inputs: dict[str, Any] = {}
        for modality, loader in self.loaders.items():
            raw = rec.get(modality)
            if raw is None:
                inputs[modality] = None
                continue
            # Text records carry the string inline; image/audio carry a path.
            if modality == "text":
                inputs[modality] = loader(raw)
            else:
                inputs[modality] = loader(_resolve(self.root, raw))
        return inputs, self.labels[idx]


# ---------------------------------------------------------------------------
# Shared split helper
# ---------------------------------------------------------------------------


def stratified_train_val_split(
    dataset: CsvDataset,
    val_split: float,
    seed: int,
) -> tuple[Subset, list[int], Subset, list[int]]:
    """Stratified train / val split of a `CsvDataset`.

    Falls back to a plain random split when the dataset is too small for
    stratification (fewer val samples than distinct classes).  Returns
    ``(train_subset, train_labels, val_subset, val_labels)``.
    """
    from sklearn.model_selection import train_test_split

    indices = list(range(len(dataset)))
    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=dataset.labels, random_state=seed
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=seed
        )

    return (
        Subset(dataset, train_idx),
        [dataset.labels[i] for i in train_idx],
        Subset(dataset, val_idx),
        [dataset.labels[i] for i in val_idx],
    )


# ---------------------------------------------------------------------------
# Shared loss factory
# ---------------------------------------------------------------------------


def focal_loss_from_labels(labels: list[int], cfg: ExperimentConfig) -> FocalLoss:
    """Build a `FocalLoss` with Cui-2019 effective-number class weights from a label list.

    Counts are clamped to ≥ 1 so absent classes never produce a division-by-zero
    in `effective_number_weights` — they simply receive the weight of a class
    with a single sample (the highest weight, as they are the rarest).
    """
    import torch

    from ..labels import NUM_CLASSES
    from ..losses import FocalLoss as _FocalLoss
    from ..losses import effective_number_weights

    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lab in labels:
        counts[lab] += 1
    weights = effective_number_weights(
        counts.clamp(min=1), beta=cfg.train.focal_loss.class_weight_beta
    )
    return _FocalLoss(
        gamma=cfg.train.focal_loss.gamma,
        label_smoothing=cfg.train.focal_loss.label_smoothing,
        class_weights=weights,
    )
