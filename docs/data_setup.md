# Data Setup

MEMO trains on five public emotion datasets. This document is the source of
truth for acquiring them: where each is hosted, which require accepting a
license, the expected on-disk layout under `data/`, and the label mapping each
undergoes en route to Ekman-7. Reaching the state described here is a hard
prerequisite for **Phase 8** (per-modality training).

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
├── affectnet/      # image — AffectNet-7 (license-gated)
├── goemotions/     # text  — GoEmotions
├── ravdess/        # audio — RAVDESS
└── cremad/         # audio — CREMA-D
```

## Datasets

### FER2013 — image · `auto`
- **Source**: https://www.kaggle.com/datasets/msambare/fer2013
- **Acquire**: Kaggle download (needs the `kaggle` CLI and an API token). Unzip
  into `data/fer2013/`.
- **Format**: 48×48 grayscale faces, 7 classes already in Ekman order.
- **Label mapping**: `labels.remap_fer2013` — identity (kept explicit so a
  reorder of `EkmanEmotion` surfaces here).

### AffectNet-7 — image · `manual` (license-gated)
- **Source**: http://mohammadmahoor.com/affectnet/
- **Acquire**: Submit the access request form. Once granted, place the 7-class
  split under `data/affectnet/`. **Not auto-downloadable** — the script only
  prints the request URL and target path.
- **Label mapping**: `labels.remap_affectnet7` — native order
  `0 neutral, 1 happy, 2 sad, 3 surprise, 4 fear, 5 disgust, 6 anger`
  → Ekman-7 (contempt from AffectNet-8 is dropped).

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
  clips). Unzip into `data/ravdess/`. Use `--k-fold` when training (RAVDESS is
  small).
- **Label mapping**: `labels.remap_ravdess` — the 3rd filename field; `calm`
  (02) collapses into `neutral` (no Ekman class for it).

### CREMA-D — audio · `auto`
- **Source**: https://github.com/CheyneyComputerScience/CREMA-D
- **Acquire**: Clone the repo (or download `AudioWAV/`) into `data/cremad/`
  (7442 clips).
- **Label mapping**: `labels.remap_cremad` — the 3-letter emotion code. CREMA-D
  has **no surprise class**.

## Manifest formats the adapters expect

`training/datasets.py` provides two adapters:

- **`CsvDataset`** (single-modality): a CSV with a value column (a file path or
  inline text) and a label column. The native label is mapped to `EkmanEmotion`
  via a `remap` callable; relative paths resolve against an optional `root`.
- **`JsonlDataset`** (aligned multimodal): one JSON object per line —
  `{"id", "image"?, "text"?, "audio"?, "label", "slices"?}`. A missing modality
  key yields `None` for that modality (the pipeline simply drops it). The
  optional `slices` dict feeds the Phase 12 fairness audit.
