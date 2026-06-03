# Architecture

System design for `memo`: how the three modality encoders, the fusion layer, and
the training pipeline fit together, and why each piece is built the way it is. The
equations live in [math.md](math.md).

## Problem

Predict one of seven Ekman emotions from a face image, an utterance transcript, a
speech clip, or any non-empty combination of the three, while degrading gracefully
when a modality is missing or unreliable. The whole pipeline targets CPU/edge
inference, so every component is chosen for a good accuracy-per-FLOP trade-off
rather than peak accuracy.

## Pipeline

```
            ┌────────────────────────────────────────────────────────────┐
            │              Raw inputs (any subset, ≥1 required)           │
            │   image: np.ndarray │ text: str │ audio: np.ndarray         │
            └────────────────────────────────────────────────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
            face preprocessing    text tokenizer       audio preprocessing
            MediaPipe → align       (MiniLM)           resample 16k → log-mel
              → 112×112                                     (64 × T)
                    │                    │                    │
                    ▼                    ▼                    ▼
            MobileNetV3-Small       MiniLM-L6 (frozen)    log-mel CRNN
            + 7-way head            + 2-layer MLP head    CNN→BiGRU→attention
            ~2.5M params            ~22M + 50K            ~0.5M params
                    │                    │                    │
                    └──────────► per-modality logits z_i ∈ ℝ⁷ ◄──────────┘
                                         │
                                         ▼
                            LateFusion (7 scalars)
                  temperature-scale → confidence-gate → renormalize → fuse
                                         │
                                         ▼
                                  EmotionPrediction
                    label · probs · per-modality probs · confidences
                    · gate weights · used modalities · abstained
```

`LateFusion.fuse` is the only component that reasons about which modalities are
present. The pipeline raises if all inputs are missing; if face detection fails on
a supplied image, that modality is dropped silently and the rest proceed.

## Modality encoders

| Modality | Encoder | Trainable | Total | Disk (FP32 / INT8) |
|---|---|---:|---:|---|
| Image | `mobilenet_v3_small` (ImageNet pretrained) + projection + 7-way head | ~0.5M | ~2.5M | ~10 / ~3 MB |
| Text | `all-MiniLM-L6-v2` (frozen) + 2-layer MLP head | ~50K | ~22M | ~80 / ~22 MB |
| Audio | log-mel → 3-block 1D CNN → BiGRU(128) → attention-pooled head | ~0.5M | ~0.5M | ~2 / ~0.6 MB |

Each encoder emits raw 7-way logits; softmax and temperature scaling happen in the
fusion layer, not the encoder.

**Image — MobileNetV3-Small, pretrained.** An edge-optimized CNN whose ImageNet
features transfer well to facial expression. The pretrained weights are loaded
explicitly so they can't be dropped by accident, which would be the single largest
accuracy regression here. Larger backbones (ViT/Swin) are ~10× slower on CPU for
marginal gain at this scale.

**Text — frozen MiniLM-L6 + small head.** MiniLM is a distilled sentence encoder,
roughly a third the weight of DistilBERT at comparable sentence-level quality.
Freezing the backbone and training only a ~50K-parameter head keeps text training
to minutes and the deployed model small. Sentence embeddings use attention-masked
mean pooling over contextual token embeddings, which preserves word order and
pretrained semantics.

**Audio — log-mel CRNN.** A 1D CNN extracts local time-frequency structure; a
single BiGRU layer captures utterance-scale prosody (pitch and energy contours)
that a flat CNN can't; attention pooling weights the expressive frames. At 0.5M
parameters it is two orders of magnitude smaller than a Wav2Vec2-scale model, which
matters for the CPU latency budget.

## Fusion design

The fusion layer is the core of the system. Each modality's logits are
temperature-scaled to a distribution, and each distribution gets a confidence
score from its normalized inverse entropy: a peaked (confident) prediction scores
near 1, a uniform (uninformative) one near 0. The fused distribution is a weighted
average where each modality's weight combines a learned per-modality weight with
its confidence raised to a learned sharpness. Absent modalities are masked out
explicitly and the weights renormalize over whatever is present, so the output
depends only on the modalities that actually contributed.

This gives the layer two useful behaviours for free: a "present but garbage"
modality (silent audio, a blurry face) has high entropy and is down-weighted, and
the system can abstain when no modality is confident enough. The whole layer is
just seven scalars — three temperatures, three weights, and one sharpness — so it
is cheap to fit and easy to interpret. Equations are in [math.md](math.md).

The decisive design choice is **how it is trained** (below): fitting the gate while
randomly dropping modalities, so it is honest about every modality subset rather
than only the all-present case.

## Training

Two stages, both required.

**Stage 1 — per-modality encoders, trained independently** on their own datasets
(see [data_setup.md](data_setup.md)). Each uses AdamW with separate backbone and
head learning rates, OneCycle scheduling, a short backbone-freeze warm-up, gradient
clipping, EMA weights, and early stopping on validation macro-F1. The loss is a
focal loss with label smoothing and effective-number class weights, paired with a
class-balanced sampler, since the emotion datasets are heavily skewed. Image and
audio use light augmentation (RandAugment / SpecAugment and friends); the frozen
text backbone needs none.

**Stage 2 — fusion calibration.** The encoders are frozen and only the seven fusion
scalars train, minimizing negative log-likelihood on an aligned validation set
while each modality is dropped independently per sample. Because the encoders are
frozen, their logits are constant and can be precomputed once, so this stage runs
in seconds. The result is a gate that performs across all seven modality subsets at
once, which is exactly the graceful-degradation property the project is about.

**Audio knowledge distillation (optional).** A frozen Wav2Vec2-Base teacher (plus a
small linear probe) supervises the CRNN student with soft targets, blended with the
hard focal loss. The teacher's logits are precomputed and cached to disk so it runs
only once, never every epoch, and it is never exported — the deployed audio model
stays at 0.5M parameters while closing most of the accuracy gap to the 95M-parameter
teacher.

**Modality dropout** is per-sample (not per-batch, which would lose within-batch
diversity) and asymmetric (text, usually the strongest signal, drops at half the
rate). Training is reproducible: seeds are set across `random`, NumPy, and torch,
deterministic algorithms are enabled, and every run writes a manifest with the
config, git SHA, and library versions.

## Evaluation

`memo evaluate` produces a markdown + JSON report:

- Headline metrics (macro-F1, weighted-F1, UAR, accuracy) plus calibration (ECE,
  Brier) for the fused output and each modality.
- A modality ablation over all seven subsets, with a gate-on-vs-off comparison that
  isolates what the confidence gating actually buys.
- A robustness sweep: fused macro-F1 as the per-sample modality-dropout rate
  increases, measured against the all-present baseline.

`memo benchmark` records per-encoder and end-to-end p95 latency, parameter counts,
MACs (via `fvcore`), and peak memory. Metrics are implemented directly and
cross-checked against scikit-learn and hand-computed fixtures.

## Engineering

- **ONNX export.** Each encoder exports independently to ONNX FP32 and dynamic
  INT8, with a parity check against PyTorch (FP32 within 1e-4 MAE, INT8 within
  5e-2). Audio uses a dynamic time axis; the fusion layer stays in NumPy at runtime.
- **Config.** Typed dataclasses with a YAML loader; unknown keys are dropped so old
  configs keep loading as the schema evolves.
- **Safety.** Every `torch.load` uses `weights_only=True`; `predict` runs under
  `torch.no_grad`.
- **CI.** Lint, format check, type check, the test suite, and an ONNX-parity smoke
  test run on every pull request.

## Project layout

```
src/memo/
├── labels.py · types.py · config.py · seed.py · logging.py
├── encoders/        base · image · text · audio
├── preprocessing/   face · text · audio
├── augment/         image · audio
├── data/            ravdess_aligned  (audiovisual → aligned trimodal JSONL)
├── losses.py        FocalLoss + label smoothing · KDLoss · class weights
├── fusion.py        LateFusion (7 scalars) + abstention + presence mask
├── pipeline.py      MultimodalEmotionPipeline.predict
├── export.py        ONNX FP32 + dynamic INT8 + parity
├── training/        trainer · datasets · samplers · modality_dropout · manifest
│                    train_{image,text,audio} · distill · calibrate_fusion
├── eval/            metrics · robustness · evaluate · benchmark
└── cli.py           predict · train · calibrate · evaluate · benchmark · export
```

## Design decisions rejected

Alternatives considered and turned down, with the reason:

| Rejected | Reason |
|---|---|
| DistilBERT for text | ~3× heavier than MiniLM-L6 for marginal sentence-task gain. |
| `mobilenet_v3_small(weights=None)` | Discards ImageNet features; the largest avoidable accuracy loss. |
| `nn.Embedding` + mean-pool text | Bag-of-embeddings discards word order and pretrained semantics. |
| Flat MLP / 2D-CNN audio encoder | Audio is sequence data; forfeits temporal structure. |
| Pure 1D-CNN audio (no BiGRU) | Loses utterance-scale prosody. |
| Max-softmax as the confidence signal | Overconfident under miscalibration; inverse entropy is more robust. |
| Cross-attention fusion | Quadratic CPU cost; train/inference shift; needs aligned data at scale. |
| Learned "null" embeddings for absent modalities | Unnecessary when absent modalities are simply masked out; dead parameters. |
| Gate trained with all modalities always present | Creates the train/inference shift the calibration is designed to avoid. |
| Per-batch modality dropout | Loses the within-batch missing-modality diversity calibration relies on. |
| Plain cross-entropy | Emotion datasets are skewed; collapses toward majority classes. |
| End-to-end joint fine-tune | Per-modality encoders + a separately-calibrated gate are cheaper and more debuggable. |
| Heavyweight serving (FastAPI/K8s), TensorRT/CoreML, federated learning, valence-arousal output | Out of scope: this is a library with an ONNX export boundary and a categorical 7-class task. |
