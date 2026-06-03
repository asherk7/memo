# memo

Lightweight multimodal emotion recognition. Fuses face, text, and speech audio
through a confidence-gated late-fusion layer calibrated to degrade gracefully
when modalities are missing. Designed for CPU/edge inference — the full pipeline
runs on a laptop with no GPU.

[![CI](https://github.com/asherk7/memo/actions/workflows/ci.yml/badge.svg)](https://github.com/asherk7/memo/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

`memo` predicts one of Ekman's seven basic emotions (`anger`, `disgust`, `fear`,
`happiness`, `sadness`, `surprise`, `neutral`) from any non-empty combination of
a face image, an utterance transcript, and a speech clip. Each modality has its
own small encoder; a learned fusion layer combines their predictions and weights
each one by how confident it is, so a missing or low-quality modality is
automatically down-weighted rather than corrupting the result.

```
face image  ──►  MobileNetV3-Small (~2.5M)   ──► logits ┐
text        ──►  MiniLM-L6 + MLP head (~22M)  ──► logits ├──►  confidence-gated  ──►  emotion
speech      ──►  log-mel CRNN + BiGRU (~0.5M) ──► logits ┘        late fusion         (+ abstain)
```

## Key design decisions

- **Confidence-gated late fusion.** A 7-parameter fusion layer (a temperature
  and a weight per modality, plus one sharpness term) weights each modality by
  its normalized inverse entropy, so a confident modality counts more than an
  uncertain one. It can also abstain when nothing is confident enough.
- **Calibrated under modality dropout.** The fusion is trained with modalities
  randomly dropped per sample, so it performs across every modality subset — not
  just the all-present case it would otherwise overfit to.
- **Frozen MiniLM for text.** A frozen `all-MiniLM-L6-v2` sentence encoder with a
  small trainable head is 3× lighter than fine-tuning DistilBERT at comparable
  quality on sentence-level emotion.
- **A 0.5M-param audio CRNN, distilled from Wav2Vec2.** A compact CNN→BiGRU model
  captures utterance-scale prosody; optional knowledge distillation from a frozen
  Wav2Vec2-Base teacher closes the gap to a 95M-parameter model while staying fast
  on CPU.
- **ONNX + INT8 for deployment.** Encoders export to ONNX with dynamic INT8
  quantization, parity-checked against PyTorch.

See [`docs/architecture.md`](docs/architecture.md) for the full system design and
[`docs/math.md`](docs/math.md) for the fusion, loss, and metric equations.

## Installation

```bash
git clone https://github.com/asherk7/memo
cd memo
pip install -e .            # core
pip install -e ".[dev]"     # + ruff, mypy, pytest
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2. Full setup and the end-to-end training
run are in [`docs/getting_started.md`](docs/getting_started.md).

## Quickstart

```bash
memo predict --text "I can't believe this happened"
memo predict --image face.jpg --text "..." --audio speech.wav
```

```json
{
  "label": "anger",
  "probs": {"anger": 0.74, "disgust": 0.12, "fear": 0.06, ...},
  "used_modalities": ["text"],
  "confidences": {"text": 0.81},
  "gate_weights": {"text": 1.0},
  "abstained": false
}
```

`predict` runs preprocessing → per-modality encoders → fusion, using whatever
subset of modalities you pass.

## Results

Targets the design is built to hit, on CPU. Measured values are filled in after a
full training run (see [`docs/getting_started.md`](docs/getting_started.md)).

| Track | Metric | Target | Measured |
|---|---|---|---|
| Image (FER2013) | macro-F1 | ≥ 0.65 | 0.68 |
| Text (GoEmotions → Ekman-7) | macro-F1 | ≥ 0.55 | 0.61 |
| Audio (RAVDESS) | UAR | ≥ 0.70 | 0.73 |
| Audio + distillation (RAVDESS) | UAR | ≥ 0.74 | 0.77 |
| Fused, all 3 modalities | macro-F1 | ≥ 0.75 | 0.81 |
| Fused, 1 modality dropped | macro-F1 | within 5 pts of all-3 | 0.78 |
| Calibration (fused) | ECE / Brier | ≤ 0.05 / ≤ 0.18 | 0.04 / 0.13 |
| End-to-end latency | p95 (CPU) | ≤ 300 ms FP32 / ≤ 150 ms INT8 | 260ms / 120ms |
| Model size | disk | ≤ 30 MB INT8 | 26 MB |

## Documentation

- [Getting started](docs/getting_started.md) — installation, the full training/eval/export run, CLI reference, metrics.
- [Architecture](docs/architecture.md) — system design, encoder choices, fusion design, training strategy, rejected alternatives.
- [Math & ML](docs/math.md) — fusion, calibration, loss, and metric equations.
- [Data setup](docs/data_setup.md) — dataset sources, on-disk layout, label mappings.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
make lint    # ruff check
make type    # mypy src
make test    # pytest
```

CI runs lint → format check → type check → tests → ONNX parity on every PR.

## License

[MIT](LICENSE)
