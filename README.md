# memo

Lightweight multimodal emotion recognition — fusing face, text, and audio via confidence-gated late fusion calibrated under modality dropout. CPU/edge inference, ONNX export, knowledge distillation, LoRA fine-tuning.

[![CI](https://github.com/asherk7/memo/actions/workflows/ci.yml/badge.svg)](https://github.com/asherk7/memo/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

**memo** classifies one of Ekman's 7 basic emotions (`anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral`) from any combination of face image, utterance text, and speech audio. It gracefully degrades when any subset of modalities is missing or corrupted.

```
face image  ──► MobileNetV3-Small (~2.5M)  ──► z_img
text        ──► MiniLM-L6 + MLP (~22M)     ──► z_txt  ──► confidence-gated ──► emotion
audio       ──► log-mel CRNN + BiGRU (~0.5M) ──► z_aud      late fusion
```

**Key design decisions:**

- **Confidence-gated late fusion** — 7 learned scalars (temperature $T_i$, weight $w_i$, sharpness $\gamma$) calibrated under modality dropout across all $2^3 - 1 = 7$ modality subsets. "Present but garbage" modalities (silent audio, blurry face) are automatically down-weighted via normalized inverse entropy.
- **MiniLM-L6 over DistilBERT** — 3× lighter, ~95% of downstream quality on sentence-level emotion tasks. Optional LoRA r=8 adapters (`--lora`) for parameter-efficient fine-tuning when the frozen head plateaus.
- **CRNN + BiGRU + attention pooling** for audio — captures utterance-scale prosody patterns that a flat CNN misses, at 0.5M params. Optional knowledge distillation from a frozen Wav2Vec2-Base teacher (`--distill`).
- **Modality-dropout calibration** — the single most important training step; teaches the fusion to perform honestly across all modality subsets, not just the all-present case.

---

## Quickstart

```bash
# Install
pip install -e ".[demo]"

# Predict (any subset of modalities)
memo predict --image face.jpg --text "I can't believe this happened" --audio speech.wav

# Text only
memo predict --text "I am absolutely furious right now"
```

Example output:

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

---

## Installation

```bash
git clone https://github.com/asherk7/memo
cd memo
pip install -e .           # core
pip install -e ".[demo]"   # + Gradio demo
pip install -e ".[dev]"    # + ruff, mypy, pytest
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.2

---

## Training

memo uses a three-stage training schedule. Stages 1 and 3 are mandatory; stage 2 requires aligned multimodal data.

### Stage 1 — Per-modality encoders (independent)

Train each encoder on its own single-modality dataset:

```bash
memo train image --data data/affectnet/ --epochs 20 --out checkpoints/image.pt
memo train text  --data data/goemotions/ --epochs 15 --out checkpoints/text.pt
memo train audio --data data/ravdess+cremad/ --epochs 20 --out checkpoints/audio.pt

# With LoRA fine-tuning on text
memo train text --data data/goemotions/ --lora --out checkpoints/text_lora.pt

# With knowledge distillation for audio (Wav2Vec2-Base → CRNN)
memo train audio --data data/ravdess+cremad/ --distill --out checkpoints/audio_kd.pt
```

Recommended datasets: FER2013 / AffectNet-7 (image), GoEmotions → Ekman-7 (text), RAVDESS + CREMA-D (audio).

### Stage 2 — Optional joint fine-tune (requires aligned data)

```bash
memo train joint \
  --aligned-train data/aligned/train.jsonl \
  --image-ckpt checkpoints/image.pt \
  --text-ckpt  checkpoints/text.pt \
  --audio-ckpt checkpoints/audio.pt \
  --out checkpoints/joint.pt
```

Aligned JSONL format (any modality field may be absent):

```json
{"id": "abc", "image": "img/abc.jpg", "text": "...", "audio": "wav/abc.wav", "label": "happiness"}
```

### Stage 3 — Fusion calibration (always run)

```bash
memo calibrate \
  --aligned-val data/aligned/val.jsonl \
  --image-ckpt checkpoints/image.pt \
  --text-ckpt  checkpoints/text.pt \
  --audio-ckpt checkpoints/audio.pt \
  --out checkpoints/fusion.pt
```

---

## Evaluation

```bash
memo evaluate \
  --aligned-test data/aligned/test.jsonl \
  --fusion-ckpt  checkpoints/fusion.pt \
  --out runs/eval-$(date +%Y%m%d)/

# With cross-dataset generalization and fairness audit
memo evaluate ... --cross-dataset --fairness
```

Results are written to `runs/<id>/` and summarized in `docs/results.md`. Primary metrics: macro-F1, UAR, ECE (15-bin), Brier score. All reported with bootstrap 95% CIs across all 7 modality subsets.

**Target metrics (FP32, CPU):**

| Track | Dataset | Metric | Target |
|---|---|---|---|
| Image only | FER2013 val | macro-F1 | ≥ 0.65 |
| Text only | GoEmotions → Ekman-7 | macro-F1 | ≥ 0.55 |
| Audio only | RAVDESS (5-fold) | UAR | ≥ 0.70 |
| Audio + KD | RAVDESS | UAR | ≥ 0.74 |
| **Fused (all 3)** | aligned val | **macro-F1** | **≥ 0.75** |
| Fused — 1 modality dropped | same | macro-F1 | ≥ within 5 pts of all-3 |
| End-to-end pipeline | CPU (p95) | latency | ≤ 300 ms FP32 / ≤ 150 ms INT8 |
| Encoders combined | disk | size | ≤ 100 MB FP32 / ≤ 30 MB INT8 |
| Fusion | ECE (15-bin) | calibration | ≤ 0.05 |

---

## Export

```bash
# ONNX FP32
memo export --fusion-ckpt checkpoints/fusion.pt --out onnx/

# ONNX INT8 (dynamic quantization, ~3× smaller)
memo export --fusion-ckpt checkpoints/fusion.pt --out onnx/ --quantize int8

# Benchmark CPU latency
memo benchmark --runs 100 --out runs/bench.json
```

ONNX parity is enforced in CI: FP32 < 1e-4 MAE, INT8 < 5e-2 MAE vs PyTorch.

---

## Demo

```bash
memo demo  # launches Gradio app at http://localhost:7860
```

Image upload + text input + microphone recording → live emotion prediction with per-modality confidence breakdown.

---

## CLI Reference

```
memo predict    --image / --text / --audio       # inference (any subset)
memo train      {image,text,audio,joint}         # training stages
memo calibrate  --aligned-val                    # fusion calibration (stage 3)
memo evaluate   --aligned-test                   # evaluation harness
memo benchmark  --runs N                         # CPU latency profiling
memo export     --out onnx/ [--quantize int8]    # ONNX export
memo demo                                        # Gradio demo
```

---

## Architecture

```
            ┌────────────────────────────────────────────────────┐
            │         Raw Inputs (any subset, ≥1 required)       │
            │  image: np.ndarray | text: str | audio: np.ndarray │
            └────────────────────────────────────────────────────┘
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │  FacePreproc  │  │  TextPreproc  │  │  AudioPreproc │
           │  MediaPipe →  │  │  MiniLM tok.  │  │  16k resample │
           │  align 112×112│  │               │  │  log-mel 64   │
           └───────────────┘  └───────────────┘  └───────────────┘
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ MobileNetV3-S │  │  MiniLM-L6    │  │ log-mel CRNN  │
           │ ImageNet pre  │  │ (frozen/LoRA) │  │ CNN→BiGRU→attn│
           │ + 7-way head  │  │ + 2-layer MLP │  │ + 7-way head  │
           │ ~2.5M params  │  │ ~22M + 50K    │  │ ~0.5M params  │
           └───────────────┘  └───────────────┘  └───────────────┘
                    │                  │                  │
                z_img∈ℝ⁷          z_txt∈ℝ⁷          z_aud∈ℝ⁷
                    │                  │                  │
                    │   p_i = softmax(z_i / T_i)          │
                    │   c_i = 1 − H(p_i) / log(7)         │
                    │   α_i = softmax(w)_i · m_i · c_i^γ  │
                    └──────────┬───────┴──────────┬────────┘
                               ▼                  ▼
                    ┌──────────────────────────────────────┐
                    │  LateFusion  (7 trainable scalars)   │
                    │  calibrated under modality dropout   │
                    │  · absent modalities → m_i = 0       │
                    │  · confidence gate via c_i^γ         │
                    │  · renormalize α; weighted prob sum  │
                    │  · optional abstention if max(p) < τ │
                    └──────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  EmotionPrediction                   │
                    │  .label  .probs  .per_modality_probs │
                    │  .confidences  .gate_weights         │
                    │  .used_modalities  .abstained        │
                    └──────────────────────────────────────┘
```

Full design rationale: [`docs/architecture.md`](docs/architecture.md)

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install

make lint    # ruff check
make type    # mypy src
make test    # pytest --cov
make bench   # CPU latency
```

CI runs on every PR: lint → type → tests → coverage ≥ 85% → ONNX parity smoke.

---

## License

[MIT](LICENSE) © 2025
