# Data setup

`memo` trains on three public emotion datasets. This is the reference for
acquiring them: where each is hosted, the expected on-disk layout, and the label
mapping each undergoes en route to Ekman-7. A fourth artifact — the aligned
trimodal set used for fusion calibration — is derived from RAVDESS (see below).

Enumerate the datasets any time:

```bash
python scripts/download_data.py --dry-run
```

It prints each dataset's source URL and target path; dropping `--dry-run` prints
per-dataset acquisition instructions.

## Directory layout

All datasets live under a single data root (default `./data`):

```
data/
├── fer2013/        # image — FER2013
├── goemotions/     # text  — GoEmotions
├── ravdess/        # audio + video — RAVDESS
└── aligned/        # built from RAVDESS (frames/ + {train,val,test}.jsonl)
```

## Single-modality datasets

### FER2013 — image
- **Source**: https://www.kaggle.com/datasets/msambare/fer2013
- **Acquire**: Kaggle download (needs the `kaggle` CLI and an API token); unzip into `data/fer2013/`.
- **Format**: 48×48 grayscale faces, 7 classes already in Ekman order.
- **Label mapping**: `labels.remap_fer2013` (identity, kept explicit so a reorder of `EkmanEmotion` surfaces here).

### GoEmotions — text
- **Source**: https://huggingface.co/datasets/google-research-datasets/go_emotions
- **Acquire**: Hugging Face `datasets` (`load_dataset("go_emotions", "simplified")`), or a CSV export into `data/goemotions/`.
- **Label mapping**: `labels.remap_goemotions` collapses the 28 fine-grained labels to Ekman-7 (accepts the integer index or the string name).

### RAVDESS — audio + video
- **Source**: https://zenodo.org/records/1188976
- **Acquire**: the audio-only speech archive (`Audio_Speech_Actors_01-24.zip`, 1440 clips) trains the audio encoder. The **video-speech** archives are also needed — the aligned set extracts face frames from them.
- **Label mapping**: `labels.remap_ravdess` reads the 3rd filename field; `calm` (02) folds into `neutral`.

## Training file formats

`training/datasets.py` provides two adapters:

- **`CsvDataset`** (single-modality): a CSV with `train.csv` (and optional `val.csv`) holding a value column and a `label` column. For image/audio the value column is `path` (resolved against the data dir); for text it is the inline `text`. Labels are the dataset-native values, mapped via `--remap-from {fer2013|goemotions|ravdess|ekman7}`.
- **`JsonlDataset`** (aligned multimodal): one JSON object per line — `{"id", "image"?, "text"?, "audio"?, "label"}`. A missing modality key yields `None` for that modality (the pipeline drops it).

## Aligned trimodal set (for fusion calibration)

Fusion calibration needs samples carrying face, text, and audio with one shared
label. RAVDESS clips provide this: each is a short audiovisual recording of one of
two fixed sentences, so a single clip yields a face frame (from the video), the
transcript (text), and the audio. Build it once:

```python
from pathlib import Path
from memo.data.ravdess_aligned import build_aligned_jsonl

build_aligned_jsonl(Path("data/ravdess"), Path("data/aligned"))
```

This parses each RAVDESS filename, extracts a middle-frame face image, pairs it
with the matching audio WAV and the transcript, and writes `train/val/test.jsonl`
split by actor (no actor appears in two splits, to avoid identity leakage). The
default audio resolver looks for the `03-…​.wav` sibling of each video clip; pass a
custom `resolve_audio=` if your audio lives in a separate tree. Clips with no
matching audio are skipped with a warning.

The text channel here is intentionally weak — only two distinct sentences across
the whole corpus — which is a feature: a well-calibrated confidence gate learns to
down-weight it.
