# Data Setup

MEMO trains on three public emotion datasets. This document is the source of
truth for acquiring them: where each is hosted, the expected on-disk layout
under `data/`, and the label mapping each undergoes en route to Ekman-7.
Reaching the state described here is a hard prerequisite for **Phase 8**
(per-modality training).

Run the enumeration any time:

```bash
python scripts/download_data.py --dry-run
```

It prints each dataset's source URL, target path, and license status
(`auto` = publicly downloadable, `manual` = request access first). Dropping
`--dry-run` prints per-dataset acquisition instructions.

## Expected directory layout

All datasets live under a single data root (default `./data`, configurable via
`paths.data` in `configs/default.yaml` or `--data-root`):

```
data/
├── fer2013/        # image — FER2013
├── goemotions/     # text  — GoEmotions
└── ravdess/        # audio — RAVDESS
```

## Datasets

### FER2013 — image · `auto`
- **Source**: https://www.kaggle.com/datasets/msambare/fer2013
- **Acquire**: Kaggle download (needs the `kaggle` CLI and an API token). Unzip
  into `data/fer2013/`.
- **Format**: 48×48 grayscale faces, 7 classes already in Ekman order.
- **Label mapping**: `labels.remap_fer2013` — identity (kept explicit so a
  reorder of `EkmanEmotion` surfaces here).

### GoEmotions — text · `auto`
- **Source**: https://huggingface.co/datasets/google-research-datasets/go_emotions
- **Acquire**: Hugging Face `datasets` (`load_dataset("go_emotions", "simplified")`)
  or the CSV export into `data/goemotions/`.
- **Label mapping**: `labels.remap_goemotions` — the 28 fine-grained labels
  collapse to Ekman-7 via the standard categorical mapping (accepts the integer
  index or the string name).

### RAVDESS — audio · `auto`
- **Source**: https://zenodo.org/records/1188976
- **Acquire**: Direct Zenodo download of `Audio_Speech_Actors_01-24.zip` (1440
  clips). Unzip into `data/ravdess/`.
- **Label mapping**: `labels.remap_ravdess` — the 3rd filename field; `calm`
  (02) collapses into `neutral` (no Ekman class for it).

## Manifest formats the adapters expect

`training/datasets.py` provides two adapters:

- **`CsvDataset`** (single-modality): a CSV with a value column (a file path or
  inline text) and a label column. The native label is mapped to `EkmanEmotion`
  via a `remap` callable; relative paths resolve against an optional `root`.
- **`JsonlDataset`** (aligned multimodal): one JSON object per line —
  `{"id", "image"?, "text"?, "audio"?, "label"}`. A missing modality key yields
  `None` for that modality (the pipeline simply drops it).
