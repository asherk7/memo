# Getting started

Installation, the end-to-end training run, the CLI, and the metrics to expect.

## Installation

```bash
git clone https://github.com/asherk7/memo
cd memo

python -m venv .venv && source .venv/bin/activate
pip install -e .            # core library + CLI
pip install -e ".[dev]"     # + ruff, mypy, pytest
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2. The library is CPU-only by default; no
GPU is needed for training or inference at this scale.

## Inference

`memo predict` runs preprocessing, the per-modality encoders, and the fusion over
whatever subset of modalities you pass, and prints a JSON prediction:

```bash
memo predict --text "I am absolutely furious right now"
memo predict --image face.jpg --text "..." --audio speech.wav --config configs/trained.yaml
```

Without trained checkpoints it runs on pretrained backbones with untrained heads
(and prints a warning) — point `--config` at a config with checkpoint paths for
real predictions.

## Training and evaluation run

The pipeline has two training stages — per-modality encoders, then fusion
calibration — plus optional audio distillation, evaluation, and ONNX export.
Everything below runs on CPU; the slowest step is the image encoder on FER2013.

### 1. Acquire the datasets

See [data_setup.md](data_setup.md) for sources, on-disk layout, and label
mappings. In short: FER2013 (image, Kaggle), GoEmotions (text, Hugging Face), and
RAVDESS audio **and video** (Zenodo). The RAVDESS video track is needed for the
aligned set in step 2.

```bash
python scripts/download_data.py --dry-run   # lists each dataset's source + target path
```

### 2. Build the aligned trimodal set

Fusion calibration needs samples that carry all three modalities with a shared
label. RAVDESS clips supply this: each is a short audiovisual recording of a fixed
sentence, so one clip yields a face frame, the transcript, and the audio.

```python
from pathlib import Path
from memo.data.ravdess_aligned import build_aligned_jsonl

build_aligned_jsonl(
    Path("data/ravdess"),     # RAVDESS video clips (+ matching WAVs)
    Path("data/aligned"),     # writes frames/ and {train,val,test}.jsonl, split by actor
)
```

### 3. Train the encoders

```bash
memo train image --data data/fer2013/   --epochs 20 --out checkpoints/image.pt
memo train text  --data data/goemotions/ --epochs 15 --out checkpoints/text.pt
memo train audio --data data/ravdess/    --epochs 20 --out checkpoints/audio.pt
```

Each run writes a checkpoint and a `runs/<id>/manifest.json` recording the config,
git SHA, library versions, and metrics.

### 4. Distill the audio encoder (optional, recommended)

```bash
export MEMO_ALLOW_HF_DOWNLOAD=1
memo train audio --data data/ravdess/ --distill --out checkpoints/audio_kd.pt
```

The first epoch runs the Wav2Vec2-Base teacher once and caches its soft targets to
disk; later epochs reuse the cache. Use `audio_kd.pt` downstream.

### 5. Wire the checkpoints into a config

`calibrate`, `evaluate`, `benchmark`, and `export` load the encoders and fusion
through a YAML config. Copy `configs/default.yaml` to `configs/trained.yaml` and
fill in the checkpoint paths:

```yaml
model:
  encoders:
    image: { checkpoint: checkpoints/image.pt }
    text:  { checkpoint: checkpoints/text.pt }
    audio: { checkpoint: checkpoints/audio_kd.pt }
  fusion: { checkpoint: checkpoints/fusion.pt }   # produced by step 6
```

### 6. Calibrate the fusion

```bash
memo calibrate --aligned-val data/aligned/val.jsonl \
               --config configs/trained.yaml \
               --modality-dropout 0.3 --out checkpoints/fusion.pt
```

The encoders stay frozen; only the seven fusion parameters train, under per-sample
modality dropout. This converges in seconds.

### 7. Evaluate and benchmark

```bash
memo evaluate  --aligned-test data/aligned/test.jsonl --config configs/trained.yaml --out runs/eval/
memo benchmark --config configs/trained.yaml --runs 100 --out runs/bench.json
```

`evaluate` writes `report.{json,md}`: headline metrics, a per-modality-subset
ablation, the gate-on-vs-off comparison, and the modality-dropout robustness
sweep. `benchmark` records per-encoder latency, parameter counts, MACs, and peak
memory.

### 8. Export to ONNX

```bash
memo export --config configs/trained.yaml --out onnx/ --quantize int8
```

Exports each encoder to ONNX FP32 and dynamic INT8 with a parity check against
PyTorch (FP32 within 1e-4 MAE, INT8 within 5e-2), plus an `onnx_export.json`
summary of sizes.

## CLI reference

```
memo predict    [--image f] [--text s] [--audio f] [--config y]   inference → JSON
memo train      {image,text,audio} --data <dir> [--distill]       per-modality training
memo calibrate  --aligned-val v.jsonl --config y                  fusion calibration
memo evaluate   --aligned-test t.jsonl --config y                 metrics + ablations report
memo benchmark  --config y --runs N                               CPU latency / MACs / memory
memo export     --config y [--quantize int8]                      ONNX export + parity
```

Every command prints one JSON object to stdout. `memo --help` and
`memo <cmd> --help` list options.

## Metrics and targets

Targets the design aims for on CPU; fill in the measured column after a run.

| Track | Metric | Target | Measured |
|---|---|---|---|
| Image only (FER2013 val) | macro-F1 | ≥ 0.65 | _TBD_ |
| Text only (GoEmotions → Ekman-7) | macro-F1 | ≥ 0.55 | _TBD_ |
| Audio only (RAVDESS) | UAR | ≥ 0.70 | _TBD_ |
| Audio + distillation (RAVDESS) | UAR | ≥ 0.74 | _TBD_ |
| Fused, all 3 modalities | macro-F1 | ≥ 0.75 | _TBD_ |
| Fused, one modality dropped | macro-F1 | within 5 pts of all-3 | _TBD_ |
| Fused calibration | ECE (15-bin) / Brier | ≤ 0.05 / ≤ 0.18 | _TBD_ |
| End-to-end latency | p95 (CPU) | ≤ 300 ms FP32 / ≤ 150 ms INT8 | _TBD_ |
| Combined model size | disk | ≤ 30 MB INT8 | _TBD_ |

UAR is unweighted average recall (the speech-emotion standard); ECE and Brier
measure calibration. See [math.md](math.md) for the metric definitions.

## Development

```bash
pre-commit install
make lint    # ruff check
make format  # ruff format
make type    # mypy src
make test    # pytest
```

CI runs lint, format check, type check, the test suite, and an ONNX-parity smoke
test on every pull request.
