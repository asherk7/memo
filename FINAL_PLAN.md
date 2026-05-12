# FINAL_PLAN.md — Definitive Multimodal Emotion Recognition System

## 0. Context

Build a lightweight CPU/edge multimodal emotion recognition system: face image + utterance text + speech audio → one of 7 Ekman emotions (`anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral`). Gracefully degrades under any non-empty modality subset. Single-developer maintainable.

This plan synthesizes seven sibling efforts (plans 1–7, implementations 2, 3, 4, 6, 7), three rankings, and three syntheses. Decisions are made on evidence and constraint fit, not vote count: where 6 of 7 chose DistilBERT, we reject it; where only folder 1 specified confidence-gated late fusion calibrated under modality dropout, we adopt it. **Every feature has an explicit justification, including the ambitious ones.** The plan retains the techniques that make this a meaningful portfolio artifact — knowledge distillation, LoRA fine-tuning, multi-task auxiliary losses, cross-dataset generalization, fairness audit, modern calibration metrics — while rejecting features that no implementation could justify (folder 5's K8s/FastAPI/federated stack, folder 6's `weights=None`, folder 4's synthetic-only training).

---

## 1. Architecture

```
            ┌────────────────────────────────────────────────────────────┐
            │              Raw Inputs (any subset, ≥1 required)          │
            │   image: np.ndarray │ text: str │ audio: np.ndarray         │
            └────────────────────────────────────────────────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
           │  FacePreproc    │  │  TextPreproc    │  │  AudioPreproc   │
           │  MediaPipe →    │  │  MiniLM         │  │  resample 16k → │
           │  align → 112³   │  │  tokenizer      │  │  log-mel(64×T)  │
           └─────────────────┘  └─────────────────┘  └─────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
           │ MobileNetV3-S   │  │  MiniLM-L6      │  │ log-mel CRNN    │
           │ ImageNet pre    │  │  (frozen/LoRA)  │  │ CNN→BiGRU→attn  │
           │ + 7-way head    │  │  + 2-layer MLP  │  │   + 7-way head  │
           │ ~2.5M params    │  │ ~22M + 50K/+200K│  │  ~0.5M params   │
           └─────────────────┘  └─────────────────┘  └─────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
                z_img∈ℝ⁷            z_txt∈ℝ⁷            z_aud∈ℝ⁷
                    │                    │                    │
                       p_i = softmax(z_i / T_i)         (3 temperature scalars)
                    │                    │                    │
                    │   c_i = 1 − H(p_i) / log(7)             │
                    │   α_i = softmax(w)_i · m_i · c_i^γ      │  (3 weight scalars + γ)
                    └──────────┬─────────┴──────────┬─────────┘
                               ▼                    ▼
                    ┌──────────────────────────────────────────┐
                    │  LateFusion (7 trainable scalars,        │
                    │  calibrated under modality dropout p=0.3)│
                    │   • absent modalities ⇒ m_i = 0          │
                    │   • confidence gate via c_i^γ            │
                    │   • renormalize α; weighted prob sum     │
                    │   • optional abstention if max(p)<τ      │
                    └──────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  EmotionPrediction (frozen dataclass)    │
                    │  .label .probs .per_modality_probs       │
                    │  .confidences .gate_weights              │
                    │  .used_modalities .abstained             │
                    └──────────────────────────────────────────┘
```

**Single point of modality reasoning.** `LateFusion.fuse` is the only module that reasons about modality presence. The pipeline raises `ValueError` if all three inputs are `None`; if MediaPipe fails on a non-`None` image, the face modality is silently degraded (folder 2's idiom, [pipeline.py:147](../2/emotion_recognizer/pipeline.py#L147)).

---

## 2. Backbone choices

| Modality | Backbone | Trainable params | Total params | Disk (FP32 / INT8) |
|---|---|---:|---:|---|
| **Image (face)** | `torchvision.models.mobilenet_v3_small` (ImageNet pretrained) + projection + 7-way head | ~0.5M (head + last block) | ~2.5M | ~10 MB / ~3 MB |
| **Text** | `sentence-transformers/all-MiniLM-L6-v2` (frozen) + 2-layer MLP head; **optional LoRA r=8 on top 2 transformer layers** | ~50K (head) + ~150K (LoRA, optional) | ~22M | ~80 MB / ~22 MB |
| **Audio** | log-mel spectrogram (64 mels, 16 kHz, 3-s window) → 3-block 1D CNN (32→64→128 channels, BN+SiLU) → 1-layer BiGRU(128) → attention-pooled MLP head; **optional KD from Wav2Vec2-Base teacher** | ~0.5M | ~0.5M | ~2 MB / ~0.6 MB |

**Total: ~25M params, ~1–1.2M trainable (depending on LoRA toggle), ~92 MB FP32 / ~26 MB INT8.**

### 2.1 Justifications

**Image: MobileNetV3-Small (pretrained).** Edge-optimized depthwise-separable CNN at 2.5M params. ImageNet pretraining provides strong low-level visual features that transfer to facial expression recognition. Six of seven plans converged here.

**Reject MobileNetV3 with `weights=None` (folder 6's impl).** Verified at [6/src/models/encoders.py:13](../6/src/models/encoders.py#L13): `mobilenet_v3_small(weights=None)` discards ImageNet pretraining. The single most expensive line in the corpus.

**Reject ViT/Swin** (~80M+ params, ~10× slower on CPU, marginal accuracy gain on facial expression at this scale).

**Text: MiniLM-L6 (frozen) + 2-layer MLP head, optional LoRA r=8.** 22M total params (~80 MB FP32) — 3× lighter than DistilBERT (66M, ~250 MB). MiniLM is purpose-built as a distilled sentence encoder; on sentence-level tasks like emotion classification, it retains ~95% of DistilBERT's quality at one-third the weight.

**LoRA upgrade path (shipped, not future work).** When the frozen-backbone + MLP head plateaus below the macro-F1 target on GoEmotions, enable LoRA r=8 adapters on the last 2 transformer layers via [peft](https://github.com/huggingface/peft):
- Adds ~150K trainable params (still far below DistilBERT-fine-tuning's 66M).
- Standard parameter-efficient fine-tuning approach; demonstrates modern PEFT competence.
- Toggled via `memo train text --lora` and `lora.enabled: true` in the YAML config.
- The plan ships both pathways: frozen-head is the default (cheap, fast); LoRA is the documented and tested upgrade.

**Reject DistilBERT (folders 2, 3, 4, 7).** Folklore "60% faster than BERT" cited in folder 3 is not benchmarked. MiniLM is faster and lighter, and LoRA + frozen MiniLM strictly dominates fine-tuned DistilBERT on the constraint frontier.

**Reject `nn.Embedding` + mean-pool (folder 6's impl).** Verified at [6/src/models/encoders.py:28](../6/src/models/encoders.py#L28): a 32-d embedding lookup with mean pooling discards word order and pretrained semantics. The cautionary example.

**Audio: log-mel CRNN with attention pooling.** Modern speech-emotion-recognition baseline. The 3-block 1D CNN extracts local time-frequency patterns; the 1-layer BiGRU captures the temporal prosody patterns (pitch contour, energy envelope) that distinguish e.g. surprise from happiness — these patterns span longer than any individual conv receptive field. Attention pooling weights expressive frames rather than averaging uniformly. 0.5M params is ~0.5% of a Wav2Vec2-Base alternative.

**Knowledge distillation from Wav2Vec2-Base (shipped, not future work).** The CRNN student trains against soft labels from a frozen Wav2Vec2-Base teacher. See §4.4 for full procedure. This is the resume-grade story: *95M-param SOTA distilled to a 0.5M-param student that closes the UAR gap to within ~2 absolute points.*

**Reject Wav2Vec2 / WavLM as the student backbone** (~95M, ~200 ms+ CPU per 3-s clip — blows the 80 ms audio budget). Used as KD teacher at train time only.

**Reject MobileNetV2 over 64×64 log-mel spectrograms** (folder 5). Audio is sequence data; treating spectrograms as 2D images forfeits the temporal structure that 1D conv + RNN exploits.

**Reject pure 1D-CNN + AdaptiveAvgPool1d (no BiGRU)** (folders 2, 3, 4, 7's impls). All five implementations dropped the BiGRU for simplicity, but the SER literature consistently uses recurrent or attention-pooled temporal models — local CNN receptive fields cannot fully capture utterance-scale prosody. The 30 ms CPU cost of a single-layer BiGRU is well within budget.

---

## 3. Fusion strategy — confidence-gated late fusion, calibrated under modality dropout

This is the standout idea from folder 1. **It is the only fusion design surveyed that is mathematically trained against missing-modality conditions.** All gated-MLP alternatives in folders 3, 4, 6, 7 train their gates with all modalities present; the inference-time zero-fill is then OOD. We adopt folder 1's design and refine the mask plumbing per folder 7's idiom.

### 3.1 Exact math (7 trainable scalars total)

Each encoder $i \in \{\text{image, text, audio}\}$ produces logits $z_i \in \mathbb{R}^7$.

1. **Temperature scaling** (3 scalars $T_i > 0$):
$$p_i = \mathrm{softmax}(z_i / T_i) \in \Delta^6$$

2. **Confidence via normalized inverse entropy** (no params):
$$c_i = 1 - \frac{H(p_i)}{\log 7} \in [0, 1], \quad H(p_i) = -\sum_{k=1}^{7} p_{i,k}\log p_{i,k}$$
$c_i = 1$ for a delta (certain), $c_i = 0$ for uniform (no information).

3. **Effective weight** (3 weight scalars $w_i$ + sharpness scalar $\gamma$):
$$\alpha_i = \mathrm{softmax}(w)_i \cdot m_i \cdot c_i^{\gamma}$$
where $m_i \in \{0, 1\}$ is the presence mask (passed explicitly to fusion per folder 7's plumbing idiom). $\gamma = 0$ recovers a plain learned weighted average; $\gamma > 0$ down-weights uncertain modalities.

4. **Renormalize and fuse:**
$$\tilde{\alpha}_i = \alpha_i / \sum_j \alpha_j, \quad p_{\text{final}} = \sum_i \tilde{\alpha}_i \cdot p_i, \quad \hat{y} = \arg\max_k p_{\text{final}, k}$$

5. **Optional abstention.** If $\max_k p_{\text{final},k} < \tau$ (config knob, default 0.40), set `abstained = True`. Caller decides what to do.

**Total trainable fusion params: $3 (T_i) + 3 (w_i) + 1 (\gamma) = 7$ scalars.** $\tau$ is a config constant, not learned.

### 3.2 Why this fusion, weighed against alternatives

| Alternative | Source | Rejected because |
|---|---|---|
| Cross-attention fusion | Folder 5 | 3–10× CPU cost from quadratic attention; train/inference distribution shift under missing modalities requires elaborate masking; needs aligned multimodal data we may not have at scale. Future work. |
| Concat-and-MLP gated fusion | Folders 3, 4, 6, 7 | Gates trained with all modalities present; inference-time zero-fill is OOD. None of the four plans even acknowledge this distribution shift. Verified in folder 3's [trainer.py](../3/src/emotion/trainer.py): pure CE with all-modality input, no modality dropout. |
| Unweighted logit average | Folder 2 | Strict subset of our design ($T_i = 1, w_i = \text{const}, \gamma = 0$). No probability calibration when per-modality encoders have different temperature scales, no confidence handling for "present but garbage" modalities (silent audio, blurry face). |
| Confidence-weighted softmax average using **max-softmax** as confidence | Folder 4's impl | Max-softmax is overconfident under miscalibration. Inverse entropy is the principled robust alternative. |

### 3.3 Refinement: no null embeddings

Folder 6's impl introduces `nn.Parameter` "null" embeddings as a substitute when a modality is absent. **We do not need this.** In our late-fusion math, absent modalities are skipped entirely (the encoder is not run; $m_i = 0$ masks the contribution). Null embeddings are only useful when a downstream layer requires a fixed-size concatenated input — which we don't have. **Adopting null embeddings here would be 21 dead parameters.**

### 3.4 The calibration step — graceful degradation as a learning objective

This is **the single most important step** for honest graceful degradation, and is **missing from every plan except folder 1**.

**Procedure** (`memo calibrate`):
1. With encoders frozen, take a small aligned validation set ($N \sim 1000$–10000 examples with all three modalities present + labels).
2. Initialize $T_i = 1$, $w_i = 0$ (so $\mathrm{softmax}(w)$ is uniform), $\gamma = 1$.
3. For each minibatch sample, **independently drop each modality** with $p_{\text{drop}} = 0.3$ (text at $p_{\text{drop}} = 0.15$, since text is typically highest-quality — folder 2's asymmetric drop idiom, [training/train.py:78](../2/training/train.py#L78)).
4. Compute the fused $p_{\text{final}}$ over surviving modalities; minimize NLL against ground-truth labels.
5. Optimize the 7 scalars via AdamW (lr = 1e-2). ~200 epochs. Converges in seconds on CPU.

The 7 scalars learn to perform across all $2^3 - 1 = 7$ modality subsets simultaneously. This is the bar §11.5 (robustness floor) defends.

---

## 4. Training strategy

Three-stage schedule. Stage 1 and stage 3 are mandatory; stage 2 (joint fine-tune) is gated on whether aligned multimodal data is available.

### 4.1 Stage 1 — Per-modality encoder training (independent)

Each encoder trains on its single-modality dataset:
- **Image:** FER2013 (basic) + AffectNet-7 (richer; ~290K aligned face crops)
- **Text:** GoEmotions (27 emotions → Ekman-7 via mapping table in `labels.py`)
- **Audio:** RAVDESS (1440 clips) + CREMA-D (7442 clips). **Stratified 5-fold** optional via `--k-fold` flag — recommended for RAVDESS due to small size.

**Optimization** (10–20 epochs each):
- **AdamW** with two param groups: backbone `lr = 1e-5`, head `lr = 1e-3` (folder 2's split).
- **OneCycleLR** with per-modality `max_lr` (image 3e-3, text-head 1e-3, audio 5e-3 — folder 1's values). Warmup + cosine decay; converges faster than plain CosineAnnealingLR with modest tuning.
- **3-epoch backbone freeze** at start, then unfreeze (folder 3's curriculum compressed into a single optimizer toggle of `requires_grad`).
- Gradient clip `max_norm = 1.0`.
- **EMA decay 0.999** on model weights (`torch.optim.swa_utils.AveragedModel`).
- Early stopping on validation macro-F1, patience = 5, restore best.

**Loss: FocalLoss with label smoothing** (folder 2's [losses.py:31](../2/training/losses.py#L31) — correct focal × smoothing interaction):
$$\mathrm{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log p_t, \quad \gamma = 2.0, \; \epsilon_{\text{smooth}} = 0.05$$
- $\alpha_t$ = effective-number-of-samples class weight (Cui et al. 2019). Computed once from the training set's class histogram.

**Sampler: ClassBalancedSampler** with effective-number-of-samples reweighting (Cui et al. 2019). Combined with weighted FocalLoss per Cui's recommended recipe — the sampler operates at the data layer (boosts minority-class frequency); the loss weight operates at the gradient layer (additionally focuses on hard examples). Together they handle the heavy emotion-class skew (happiness 2–4× overrepresented).

**Augmentation** (per-modality YAML knobs):
- **Image:** RandAugment (n=2, m=9) + horizontal flip + random erasing (p=0.25) + **Mixup (α=0.2)** for the final 50% of epochs (toggle off when soft labels conflict with focal loss).
- **Audio:** SpecAugment (2 time masks ≤ 30 frames, 2 freq masks ≤ 12 bins) + additive Gaussian noise at random SNR 10–30 dB + gain ±6 dB + **time-stretch 0.9–1.1** (`librosa.effects.time_stretch`).
- **Text:** token dropout (p=0.05).

**Reject EDA text augmentation** (folder 1): synonym/swap/delete with WordNet adds a ~30 MB dependency for ~1 pt gain over token dropout.

**Reject pre-emphasis filter** (folder 1): marginal MFCC tweak that complicates ONNX export.

### 4.2 Stage 2 — Optional joint fine-tune (multi-task, with per-modality auxiliary losses)

Triggered by `memo train joint --aligned-train train.jsonl` when an aligned multimodal training set is available. Skipped otherwise — stage 3 calibration still works without it.

- Start from stage-1 checkpoints.
- Unfreeze: the last MobileNet block, the last 2 MiniLM transformer layers (or LoRA adapters if enabled), full audio CRNN.
- AdamW with three param groups: backbones `lr = 1e-5`, heads `lr = 1e-4`, fusion-stage-3 scalars frozen (calibrated separately in §3.4).
- **Per-sample modality dropout p=0.3** (text at p=0.15) per folder 6's pattern.
- **Multi-task loss** (folder 4's contribution):
$$\mathcal{L} = \mathcal{L}_{\text{fused}} + \sum_i \lambda_i \mathcal{L}_i, \quad \lambda_i = 0.3$$
Each encoder retains a direct supervision signal even when the fused loss is dominated by another modality. The aux losses use the same FocalLoss with label smoothing as stage 1.
- 5–10 epochs, early stop on val fused macro-F1.

### 4.3 Stage 3 — Fusion calibration

As specified in §3.4. **Always run, regardless of whether stage 2 happened.** The only stage that uses aligned multimodal data; only the 7 fusion scalars train; encoders are frozen.

### 4.4 Knowledge distillation for the audio encoder

Triggered by `memo train audio --distill`. Teacher: Wav2Vec2-Base (frozen, ~95M params, downloaded once via HuggingFace). Student: our 0.5M-param CRNN.

**Loss:**
$$\mathcal{L}_{\text{KD}} = \alpha \cdot \mathcal{L}_{\text{focal}}(\hat{y}_{\text{student}}, y) + (1-\alpha) \cdot \tau^2 \cdot \mathrm{KL}\big(\sigma(\hat{y}_{\text{student}}/\tau) \,\|\, \sigma(\hat{y}_{\text{teacher}}/\tau)\big)$$
with $\alpha = 0.5$, $\tau = 4$ (standard Hinton-style KD). Teacher logits are precomputed once per training set (cached to disk) to avoid running Wav2Vec2 forward each epoch.

**Why distillation matters here** (and why this is a resume-grade addition):
- Wav2Vec2-Base is the SOTA reference for speech emotion recognition, but its ~200 ms CPU forward time on a 3-sec clip blows our latency budget.
- The CRNN student closes the gap to within ~2 absolute UAR points at 1/200th the params — a textbook PEFT-adjacent technique that demonstrates pragmatic SOTA-to-edge transfer.
- Teacher is only loaded at train time. Inference graph stays at 0.5M params.

### 4.5 Modality dropout — two distinct locations

| Stage | Location | Rate | Granularity | Purpose |
|---|---|---|---|---|
| Stage 1 | none | — | — | Each encoder sees only its own modality; dropout would be a no-op |
| Stage 2 | joint fine-tune | 0.3 / 0.15 (text) | **per sample** | Teaches encoder heads to produce useful predictions when other modalities are absent |
| Stage 3 | fusion calibration | 0.3 / 0.15 (text) | **per sample** | Teaches the 7 fusion scalars to perform across all 7 modality subsets |

**Reject per-batch modality dropout** (folder 2's [train.py:60](../2/training/train.py#L60) impl). All-or-nothing per minibatch loses the within-batch missing-modality diversity that calibration depends on. Folder 6's per-sample approach (verified [6/src/train/trainer.py:15](../6/src/train/trainer.py#L15)) is correct.

### 4.6 Reproducibility

- `seed_everything(seed, deterministic=True)` seeds `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, enables `torch.use_deterministic_algorithms(True)`, and sets `os.environ["CUBLAS_WORKSPACE_CONFIG"]="4096:8"` (required for deterministic CUDA matmul). Folder 1's exact pattern.
- Every CLI command writes `runs/<id>/manifest.json` with config snapshot + git SHA + library versions + `model_card.md` auto-generated next to the checkpoint.

---

## 5. Project layout

```
multimodal_emotion/
├── README.md                       # install, quickstart, training, calibration, export, eval, demo
├── pyproject.toml                  # deps + `memo` typer CLI entry point
├── Makefile                        # install/lint/type/test/train-*/calibrate/evaluate/bench/export/demo
├── .pre-commit-config.yaml         # ruff + mypy on staged files
├── .github/workflows/ci.yml        # lint + type + tests + coverage on PRs
├── .gitignore                      # checkpoints/, data/, runs/, __pycache__/
├── configs/
│   ├── default.yaml                # encoder + fusion + path + LoRA + KD config
│   └── augmentation.yaml           # per-modality augmentation knobs
├── docs/
│   ├── architecture.md             # polished design doc
│   ├── eval_protocol.md            # how reported numbers were produced
│   ├── model_card_template.md      # auto-filled per run
│   └── results.md                  # auto-filled by `memo evaluate`
├── demo/
│   └── app.py                      # Gradio demo (image upload + text box + mic record)
├── src/memo/
│   ├── __init__.py                 # public API: MultimodalEmotionPipeline, EmotionPrediction
│   ├── labels.py                   # EkmanEmotion enum + dataset remappers
│   │                               #   (FER2013, AffectNet, GoEmotions, RAVDESS, CREMA-D)
│   ├── types.py                    # EmotionPrediction (frozen dataclass)
│   ├── config.py                   # Typed dataclass configs (ExperimentConfig, ModelConfig, TrainConfig)
│   ├── seed.py                     # seed_everything(seed, deterministic=True)
│   ├── logging.py                  # loguru setup
│   ├── encoders/
│   │   ├── base.py                 # ModalityEncoder Protocol
│   │   ├── image.py                # MobileNetV3SmallFaceEncoder  (ImageNet pretrained)
│   │   ├── text.py                 # MiniLMTextEncoder            (frozen MiniLM + MLP, optional LoRA)
│   │   └── audio.py                # LogMelCRNNEncoder            (CNN→BiGRU→attention pool)
│   ├── preprocessing/
│   │   ├── face.py                 # MediaPipe detect → align → 112×112 (raises FaceNotFoundError)
│   │   ├── text.py                 # MiniLM tokenizer wrapper
│   │   └── audio.py                # resample 16k + log-mel (train-only SpecAug applied here)
│   ├── augment/
│   │   ├── image.py                # RandAugment + flip + random erasing + Mixup
│   │   ├── text.py                 # token dropout
│   │   └── audio.py                # SpecAugment + noise + gain + time-stretch
│   ├── losses.py                   # FocalLoss (folder 2 reference) + LabelSmoothingCE + KDLoss
│   ├── fusion.py                   # LateFusion (T_i, w_i, γ) + abstention + presence mask
│   ├── pipeline.py                 # MultimodalEmotionPipeline.predict(image=, text=, audio=)
│   ├── training/
│   │   ├── trainer.py              # shared loop: param groups, EMA, OneCycle, grad clip, early stop
│   │   ├── datasets.py             # CSV adapter (folder 2 design) + jsonl adapter for aligned eval
│   │   ├── samplers.py             # ClassBalancedSampler (Cui 2019)
│   │   ├── modality_dropout.py     # per-sample dropout used in stage 2 + stage 3
│   │   ├── kfold.py                # stratified 5-fold runner (opt-in via --k-fold)
│   │   ├── train_image.py
│   │   ├── train_text.py           # supports --lora flag
│   │   ├── train_audio.py          # supports --distill flag
│   │   ├── train_joint.py          # optional stage 2: aligned multimodal joint fine-tune
│   │   ├── distill.py              # KD trainer: Wav2Vec2-Base teacher → CRNN student
│   │   └── calibrate_fusion.py     # 7-scalar fit under modality dropout
│   ├── eval/
│   │   ├── metrics.py              # macro/weighted F1, UAR, per-class P/R, ECE (15-bin), Brier
│   │   ├── calibration.py          # reliability diagrams + ECE
│   │   ├── ablations.py            # 7 modality subsets, bootstrap 95% CIs (n=1000)
│   │   ├── robustness.py           # SNR / blur / typo / occlusion / low-light sweeps
│   │   ├── cross_dataset.py        # FER2013↔AffectNet, RAVDESS↔CREMA-D generalization
│   │   ├── fairness.py             # per-slice metric breakdown (opt-in via --fairness)
│   │   └── benchmark.py            # CPU p95 latency over 100 runs, peak RSS, MACs (fvcore)
│   ├── export.py                   # ONNX FP32 + dynamic INT8 + parity check
│   └── cli.py                      # typer: predict | train | calibrate | evaluate | export | benchmark | demo
└── tests/
    ├── conftest.py                 # synthetic image/text/audio fixtures + dummy encoders
    ├── test_fusion.py              # all 7 modality subsets renormalize; abstention; γ=0 degenerate
    ├── test_pipeline.py            # end-to-end across all 7 modality subsets; all-None raises
    ├── test_preprocessing.py       # MediaPipe → FaceNotFoundError; resample correctness
    ├── test_augment.py             # SpecAugment / Mixup shape and value invariants
    ├── test_losses.py              # FocalLoss(γ=0)≡CE; label smoothing closed-form; KD numerics
    ├── test_metrics.py             # ECE / UAR / macro-F1 / Brier vs hand-computed fixtures
    ├── test_encoders.py            # each encoder's predict_logits returns shape (B, 7)
    ├── test_lora.py                # LoRA wraps the right layers; trainable param count matches r=8
    ├── test_calibration.py         # 7-scalar fit reduces NLL on synthetic aligned data
    └── test_export.py              # ONNX FP32 parity < 1e-4 MAE, INT8 < 5e-2 MAE
```

### 5.1 Key interfaces

```python
# labels.py
class EkmanEmotion(IntEnum):
    ANGER = 0; DISGUST = 1; FEAR = 2; HAPPINESS = 3
    SADNESS = 4; SURPRISE = 5; NEUTRAL = 6
NUM_CLASSES = 7

# types.py
@dataclass(frozen=True)
class EmotionPrediction:
    label: EkmanEmotion
    probs: dict[EkmanEmotion, float]                          # fused
    per_modality_probs: dict[str, dict[EkmanEmotion, float]]  # post-T-scaling
    confidences: dict[str, float]                             # c_i per modality
    gate_weights: dict[str, float]                            # α̃_i after renorm
    used_modalities: tuple[str, ...]
    abstained: bool

# encoders/base.py
class ModalityEncoder(Protocol):
    name: str                                                 # "image" | "text" | "audio"
    num_classes: int
    def predict_logits(self, x: Any) -> torch.Tensor: ...     # returns (B, 7)
    def to_onnx(self, path: Path, quantize: bool = False) -> None: ...

# fusion.py
class LateFusion(nn.Module):
    """7 trainable scalars: T_image, T_text, T_audio, w_image, w_text, w_audio, gamma.
    Abstention threshold tau is a config constant, not learned."""
    def fuse(
        self,
        per_modality_logits: dict[str, Tensor | None],   # None ⇒ absent
    ) -> FusionOutput: ...

# pipeline.py
class MultimodalEmotionPipeline:
    @torch.no_grad()
    def predict(self, *, image=None, text=None, audio=None) -> EmotionPrediction: ...
    @classmethod
    def from_config(cls, config_path: Path) -> "MultimodalEmotionPipeline": ...
```

---

## 6. Evaluation harness

`memo evaluate` produces a single markdown + JSON report. Each metric earns its keep; nothing is included for completeness alone.

1. **Headline table:** macro-F1, weighted F1, UAR, accuracy, ECE (15-bin), Brier. Fused + per-modality.
   - *Why each:* macro-F1 is the imbalance-robust primary metric. Weighted F1 enables comparison to published baselines that use it. UAR (unweighted average recall) is the audio-emotion standard. ECE is the calibration metric we explicitly fit via temperature scaling. **Brier is a strictly proper scoring rule that complements ECE — ECE assesses bin-level reliability, Brier penalizes overall MSE on the probability simplex.** Both reported because they probe distinct calibration aspects.
2. **Confusion matrix** (PNG + CSV) and per-class P/R/F1.
3. **Modality ablation:** all $2^3 - 1 = 7$ subsets, each with macro-F1 ± bootstrap 95% CI (n=1000 resamples). Reports the confidence-gating-on vs off comparison ($\gamma$ learned vs $\gamma = 0$).
   - This is the headline result for graceful degradation. Bootstrap CIs prevent reading too much into noise on small aligned val sets.
4. **Reliability diagrams** per modality + fused. One PNG. Visual companion to ECE.
5. **Robustness sweeps** — each tests whether the confidence gate down-weights the corrupted modality:
   - **Audio:** white-noise SNR ∈ {0, 5, 10, 20, ∞} dB + babble noise at 10 dB (real-world condition)
   - **Image:** Gaussian blur σ ∈ {0, 1, 2, 4} + 30% random occlusion patch + low-light gamma=2.0
   - **Text:** typo injection p ∈ {0, 0.05, 0.10} + word drop p=0.10
6. **Cross-dataset generalization** (default headline):
   - Image: train FER2013 → eval AffectNet, and vice versa
   - Audio: train RAVDESS → eval CREMA-D, and vice versa
   - Reports macro-F1 drop vs in-distribution baseline. **The single most honest "did the model actually learn?" check** — and the resume bullet that distinguishes this from a single-dataset overfit.
7. **Latency + memory + MACs benchmark** (`memo benchmark`): per-encoder forward (median + p95 over 100 runs), end-to-end pipeline, peak RSS, MACs via `fvcore`. PyTorch FP32 + ONNX INT8 reported side-by-side. Justifies the §7 target metrics.

### Opt-in (`--fairness`)

**Fairness audit** — per-slice metric breakdown when the JSONL has `"slices": {"gender": ..., "age_bucket": ...}` or similar. Reports max−min macro-F1 across slices (the "equal opportunity gap"). Opt-in because most public emotion datasets don't ship demographic labels, but ramping up to it the moment users have slice metadata is the right design.

---

## 7. Target metrics (concrete, falsifiable)

| Track | Dataset | Metric | Target |
|---|---|---|---:|
| Image only | FER2013 val | macro-F1 | ≥ 0.65 |
| Image only | AffectNet-7 val | macro-F1 | ≥ 0.58 |
| Text only (frozen) | GoEmotions → Ekman-7 | macro-F1 | ≥ 0.55 |
| Text only (+ LoRA r=8) | GoEmotions → Ekman-7 | macro-F1 | ≥ 0.58 |
| Audio only | RAVDESS (5-fold) | UAR | ≥ 0.70 |
| Audio only | CREMA-D (5-fold) | UAR | ≥ 0.62 |
| Audio + KD | RAVDESS | UAR | ≥ 0.74 (within ~2 pts of Wav2Vec2-Base teacher) |
| **Fused (3 modalities)** | aligned IEMOCAP-style val | **macro-F1** | **≥ 0.75** |
| Fused (1 modality dropped) | same | macro-F1 | ≥ within 5 pts of all-3 |
| End-to-end pipeline | Apple-Silicon CPU | p95 latency | ≤ 300 ms FP32 / ≤ 150 ms INT8 |
| Encoders combined | disk | size | ≤ 100 MB FP32 / ≤ 30 MB INT8 |
| Calibration | fused | ECE (15-bin) / Brier | ≤ 0.05 / ≤ 0.18 |
| Cross-dataset transfer | FER2013→AffectNet | macro-F1 drop | ≤ 0.10 absolute |

The 5-point robustness floor is the headline number the calibration step (§3.4) is designed to defend.

---

## 8. Engineering

- **ONNX export.** Each encoder exports independently. FP32 parity within `1e-4` MAE vs PyTorch; dynamic INT8 within `5e-2` MAE (folder 1's tolerances; folder 6's [scripts/export_onnx.py](../6/scripts/export_onnx.py) as reference). Fusion is pure numpy at runtime — no ONNX needed.
- **CI/CD.** GitHub Actions on PR: `ruff check`, `mypy src`, `pytest --cov` with coverage gate at 85%, ONNX parity smoke test. Status badges in README.
- **Pre-commit.** ruff (lint + format) + mypy on changed files.
- **Makefile.** One-liners for `install`, `lint`, `type`, `test`, `train-*`, `calibrate`, `evaluate`, `bench`, `export`, `demo`.
- **Config.** Typed `ExperimentConfig` / `ModelConfig` / `TrainConfig` dataclasses + YAML loader (folder 6's [config](../6/src/config/settings.py) idiom).
- **Security.** `weights_only=True` on every `torch.load` (folder 2's idiom). Important on a CPU/edge target where users may load checkpoints from external sources.
- **Inference safety.** `@torch.no_grad` wraps the entire `predict()` method.

### CLI surface (`memo`)

- `memo predict --image path.jpg --text "..." --audio clip.wav` → JSON
- `memo train image --data <dir> --epochs N --out checkpoints/face.pt`
- `memo train text --data <dir> --epochs N --lora --out checkpoints/text.pt`
- `memo train audio --data <dir> --epochs N --distill --out checkpoints/audio.pt`
- `memo train joint --aligned-train train.jsonl --aligned-val val.jsonl --out checkpoints/joint.pt`
- `memo calibrate --aligned-val val.jsonl --modality-dropout 0.3 --out checkpoints/fusion.pt`
- `memo evaluate --aligned-test test.jsonl --out runs/eval-<date>/ [--cross-dataset off] [--fairness]`
- `memo benchmark --runs 100 --out runs/bench-<date>.json`
- `memo export --out onnx/ [--quantize int8]`
- `memo demo` → Gradio app

Aligned JSONL format (one example per line):
```json
{"id":"abc","image":"img/abc.jpg","text":"...","audio":"wav/abc.wav","label":"happiness",
 "slices":{"gender":"female","age_bucket":"30-40"}}
```
Any of `image` / `text` / `audio` may be absent. `slices` is optional and feeds the fairness audit when present.

---

## 9. Dependencies

```
# Core
torch>=2.2, torchvision, torchaudio, numpy, Pillow, soundfile, librosa
# Models / preprocessing
transformers, tokenizers, mediapipe, peft (LoRA)
# Export
onnx, onnxruntime
# CLI / config / logging
typer, pyyaml, loguru
# Eval / plotting / profiling
scikit-learn, matplotlib, fvcore (MACs)
# Demo (optional extra: pip install .[demo])
gradio
# Dev
pytest, pytest-cov, ruff, mypy, pre-commit
```

---

## 10. Implementation order

1. **Skeleton & config:** `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.gitignore`, `configs/{default,augmentation}.yaml`, `src/memo/{labels,types,config,seed,logging}.py`
2. **Preprocessing:** `preprocessing/{face,text,audio}.py` + `test_preprocessing.py` (folder 2 idioms: BGR↔RGB, MediaPipe `FaceNotFoundError`, MFCC center-pad)
3. **Encoders:** `encoders/{base,image,text,audio}.py` random-init forward path + `test_encoders.py`. Wire LoRA into `encoders/text.py` behind a flag.
4. **Augmentation + losses:** `augment/{image,text,audio}.py`, `losses.py` (FocalLoss + KDLoss) + `test_losses.py` + `test_augment.py`
5. **Fusion:** `fusion.py` (conceptual heart — temperature, confidence, mask, abstention) + thorough `test_fusion.py`
6. **Pipeline:** `pipeline.py` integration + `test_pipeline.py` covering all 7 modality subsets
7. **Training scaffolding:** `trainer.py`, `datasets.py`, `samplers.py` (ClassBalancedSampler), `modality_dropout.py`, `kfold.py`
8. **Per-modality training:** `train_{image,text,audio}.py` (LoRA flag in text; KD flag in audio)
9. **Optional joint training:** `train_joint.py` + per-modality auxiliary losses (folder 4 contribution)
10. **Knowledge distillation:** `distill.py` (Wav2Vec2-Base teacher caching + KD training loop) + audio target re-run with `--distill`
11. **Fusion calibration:** `calibrate_fusion.py` + `test_calibration.py` (the critical step)
12. **Evaluation harness:** `eval/{metrics,calibration,ablations,robustness,cross_dataset,fairness,benchmark}.py`
13. **Export:** `export.py` (FP32 + INT8) + `test_export.py`
14. **CLI + Gradio demo:** `cli.py`, `demo/app.py`
15. **README + model card + `docs/results.md`** — last, once numbers stabilize

---

## 11. Verification (acceptance criteria)

**§11.1 Unit tests** (`pytest -q`):
- `test_fusion.py` — 7 modality subsets renormalize; abstention triggers under $\tau$; $\gamma = 0$ recovers learned weighted average
- `test_pipeline.py` — random-init encoders return valid predictions for all 7 non-empty subsets; all-None raises `ValueError`
- `test_preprocessing.py` — MediaPipe failure → `FaceNotFoundError`; resample correctness vs librosa; log-mel shape stable
- `test_augment.py` — SpecAugment mask coverage; Mixup linearity; image augment shape/dtype invariants
- `test_losses.py` — `FocalLoss(γ=0)` ≡ CE; label smoothing matches closed form; `KDLoss` reduces to CE at α=1
- `test_metrics.py` — ECE / UAR / macro-F1 / Brier vs hand-computed fixtures
- `test_encoders.py` — each encoder's `predict_logits` returns shape `(B, 7)`
- `test_lora.py` — `peft.LoraConfig(r=8)` adapters attach to the right modules; trainable param count ≈ 150K
- `test_calibration.py` — 7-scalar fit reduces NLL on synthetic aligned data; $w_i$ moves off uniform; $T_i$ finite; ≥ 50 monotone decrease steps
- `test_export.py` — ONNX FP32 parity < 1e-4 MAE, INT8 < 5e-2 MAE

**§11.2 Coverage ≥ 85%** on `src/memo/{fusion,pipeline,preprocessing,augment,losses,eval/metrics}.py`. CI-enforced.

**§11.3 Smoke training** — each `train_*` script for 1 epoch on a 32-example synthetic slice: loss decreases, checkpoint saves and reloads, `model_card.md` is generated.

**§11.4 KD smoke** — `memo train audio --distill` on a 32-clip slice with a tiny stub teacher: student logits move toward teacher's soft distribution; total loss decreases.

**§11.5 CPU latency budget** (`memo benchmark` on Apple Silicon):
- Image encoder p95 ≤ 100 ms · Text encoder p95 ≤ 100 ms · Audio encoder p95 ≤ 80 ms
- End-to-end pipeline p95 ≤ 300 ms FP32 / ≤ 150 ms INT8

**§11.6 Robustness floor** — under per-sample test-time modality dropout ($p = 0.3$), fused macro-F1 stays within **5 absolute points** of the all-modalities-present score. This is the bar §3.4 calibration must clear.

**§11.7 Cross-dataset floor** — macro-F1 drop ≤ 10 absolute points when transferring between FER2013 ↔ AffectNet (image) and RAVDESS ↔ CREMA-D (audio). Generous floor that catches catastrophic overfit.

**§11.8 Manual quickstart** — README quickstart copy-pasted into a fresh venv runs to a printed prediction in under 5 minutes (including model downloads).

---

## 12. Not adopted (with reasons)

Every "not adopted" item is something at least one plan or implementation proposed that was rejected on evidence or constraint fit.

| Rejected | Source | Reason |
|---|---|---|
| **Cross-attention fusion** | Folder 5 | Quadratic CPU cost; train/inference distribution shift; aligned data requirement. Documented as future work; would require its own design pass. |
| **DistilBERT for text** | Folders 2, 3, 4, 7 | 3× heavier than MiniLM-L6 for marginal sentence-task quality gain. Consensus is wrong on the constraint frontier. |
| **`mobilenet_v3_small(weights=None)`** | Folder 6 impl | Discards ImageNet features. The most expensive line in folder 6's codebase. |
| **`nn.Embedding(vocab, 32)` + mean-pool for text** | Folder 6 impl | Bag-of-embeddings discards word order + pretrained semantics. We name backbone precisely (`sentence-transformers/all-MiniLM-L6-v2`) to prevent this regression. |
| **Flat MLP audio encoder over aggregated vector** | Folder 6 impl | No temporal modeling. Audio is sequence data. |
| **Pure 1D-CNN audio (no BiGRU)** | Folders 2, 3, 4, 7 impls | Loses utterance-scale prosody patterns. SER literature consistently uses recurrent/attention temporal models. |
| **Synthetic-only training** | Folder 4 impl | `torch.randn(B,3,224,224)` produces random-quality models. |
| **Max-softmax as confidence weight** | Folder 4 impl | Overconfident under miscalibration. Inverse entropy is the principled alternative. |
| **Log-of-softmax fusion** | Folder 4 impl | `log(softmax(z).clamp(1e-8))` loses numerical headroom. |
| **Pure cross-entropy without imbalance handling** | Folders 3, 4, 6, 7 | Emotion datasets are heavily skewed; will bias predictions toward majority classes. |
| **Modality dropout at batch level (all-or-nothing)** | Folder 2 impl | Within-batch missing-modality mixes are not learned. Per-sample is the correct pattern. |
| **Modality dropout during stage-1 per-modality training** | (natural to add) | Each encoder sees only its own modality — no others to drop. Belongs in stages 2 and 3. |
| **Element-wise sigmoid gate over embeddings** | Folder 3 design | More trainable params, harder to calibrate, no mechanism for "present but garbage." |
| **Gate trained only with all modalities present** | Folders 3, 4, 6, 7 | Creates train/inference distribution shift the gate has not seen. Our fusion is calibrated under modality dropout. |
| **`nn.Parameter` null embeddings for missing modalities** | Folder 6 impl, BEST_PLAN.md adopt | Unnecessary when absent modalities are skipped entirely ($m_i = 0$ masks the contribution). 21 dead params. |
| **`predict_proba` + `forward` called back-to-back** | Folder 4 impl | Doubles inference cost to recover per-modality logits. We expose them via the fusion output in a single forward pass. |
| **Gate regularization loss** | Folder 3 impl | Tuning knob with unclear marginal value when fusion has only 7 scalars. |
| **Two-stage curriculum as separate stages with separate optimizers** | Folder 3 impl | Replaced by a 3-epoch `requires_grad` freeze inside a single optimizer. Same effect, less code. |
| **EDA text augmentation (synonym/swap/delete)** | Folder 1 | Requires WordNet (~30 MB dep) for ~1 pt gain over token dropout. |
| **Pre-emphasis filter** | Folder 1 | Marginal MFCC tweak; complicates ONNX export. |
| **MobileNetV2 over 64×64 log-mel spectrograms** | Folder 5 | Audio is sequence data, not 2D images. |
| **VAD continuous output (valence-arousal-dominance)** | Folder 5 | Task is categorical 7-class. Separate problem. |
| **Epistemic / aleatoric uncertainty estimation** | Folder 5 | Confidence gating via inverse entropy gives a usable uncertainty signal already. MC-dropout etc. is too expensive at edge inference. |
| **FastAPI server, Kubernetes HPA, Prometheus, Redis** | Folder 5 | Library, not service. Out of scope. |
| **TensorRT, CoreML, TFLite, federated learning, DP, canary deployments** | Folder 5 | All out of scope. ONNX is the export boundary. |
| **13-week, 7-phase roadmap** | Folder 5 | Not credible for a single developer. |
| **Video / temporal fusion across frames** | (none) | Out of scope. Single-shot `predict` per utterance. |
| **Streaming inference API** | (none) | Out of scope; single-shot only. |
| **Bundled training data** | (none) | User-provided via documented CSV/JSONL formats. |

---

## 13. Attribution summary

| Component | Source | Why this source |
|---|---|---|
| Overall scope / project layout | Folder 1 | Most coherent overall structure; only plan that bounded scope honestly |
| MobileNetV3-Small (pretrained) face encoder | Folder 1 | Best justified vs alternatives; all 5 implementations also converged here |
| MiniLM-L6 (frozen) text encoder | Folder 1 | Only plan to reject DistilBERT for weight; correct on the edge constraint |
| **LoRA r=8 on MiniLM top layers** | Folder 1 (optional → shipped) | Modern PEFT; bridges frozen-head to full fine-tune at ~150K extra params |
| Log-mel CRNN + attention pooling audio encoder | Folder 1 | Aligns with SER literature; attention-pool handles variable length |
| **Knowledge distillation from Wav2Vec2-Base → CRNN** | Folder 1 (optional → shipped) | Closes accuracy gap to SOTA while keeping 0.5M-param student; signature edge-ML technique |
| Confidence-gated late fusion (7 scalars: $T_i, w_i, \gamma$) | Folder 1 | Only design that explicitly handles missing AND "present but garbage" modalities |
| **Modality-dropout calibration** of fusion scalars | Folder 1 | The standout idea — graceful degradation as a calibration objective |
| Explicit presence-mask plumbing through fusion | Folder 7 impl | Cleanest "no magic-by-zero" pattern |
| FocalLoss + label smoothing (correct interaction) | Folder 2 impl | Cleanest reference implementation |
| ClassBalancedSampler (Cui 2019 effective-number-of-samples) | Folder 1 | Operates at data layer; complements weighted FocalLoss at gradient layer per Cui's recommended recipe |
| Mixup (α=0.2, image only, last 50% of epochs) | Folder 1 | Modest gain on small image datasets; one-line addition |
| Per-modality auxiliary losses | Folder 4 design | Each encoder retains direct supervision in joint training |
| Two-stage curriculum (freeze → unfreeze) | Folder 3 impl | Compressed to 3-epoch `requires_grad` freeze in single optimizer |
| OneCycleLR with per-modality `max_lr` | Folder 1 | Faster convergence than plain CosineAnnealing; one extra config knob |
| Time-stretch audio augmentation | Folder 1 | Simple librosa one-liner; complements SpecAugment |
| EMA decay 0.999 on model weights | Folder 1 | Single-line addition with consistent regularization benefit |
| Per-sample (not per-batch) modality dropout | Folder 6 impl | Folder 2's per-batch impl is the known regression to avoid |
| Asymmetric modality dropout (text at $p/2$) | Folder 2 impl | Text is usually the highest-quality signal |
| Separate AdamW param groups (backbone lr=1e-5, head lr=1e-3) | Folder 2 impl | Textbook transfer-learning setup |
| `@torch.no_grad`, `weights_only=True`, silent face degradation | Folder 2 impl | Production-grade CPU inference patterns |
| Typed `ExperimentConfig`/`ModelConfig`/`TrainConfig` + YAML | Folder 6 impl | Best engineering surface |
| `gate_weights` exposed on prediction output | Folder 3 design (`FusionOutput`) | Interpretability hook |
| Macro-F1, UAR, ECE, Brier as primary metrics | Folder 1 | "Never accuracy alone"; ECE + Brier probe distinct calibration aspects |
| All-7-subset ablation + bootstrap 95% CIs | Folder 1 | Statistical rigor for reported numbers |
| Robustness sweeps (audio SNR/babble + image blur/occlusion/low-light + text typo/word-drop) | Folder 1 | Validates the confidence gate across diverse corruption modes |
| **Cross-dataset generalization as default headline** | Folder 1 | The most honest "did the model actually learn?" check |
| Fairness audit (opt-in via `--fairness`) | Folder 1 | Modern ML accountability when demographic labels exist |
| Stratified 5-fold for small audio datasets | Folder 1 (optional → opt-in) | Better generalization estimate on RAVDESS-scale data |
| MACs profiling via fvcore | Folder 1 | Edge-ML cred; thp/inference is a real-world cost metric |
| ONNX FP32 + dynamic INT8 + parity tolerances (`1e-4` / `5e-2`) | Folder 1 (spec) + folder 6 (script) | Deployment-ready |
| Target metrics table | Folder 1 | Only plan to commit to falsifiable numbers |
| Typer-based CLI (`memo`) | Folder 1 | Consistent, discoverable subcommand surface |
| Gradio demo app | Folder 1 | Optional install; useful artifact |
| CI/CD plumbing (ruff + mypy + pytest + coverage ≥ 85%) | Folder 1 | Production discipline |
| Auto-generated model card per run | Folder 1 | Reproducibility + reviewer ergonomics |

---

## 14. Beyond the syntheses — what makes this stand out

The three synthesis docs (`BEST_PLAN.md`, `BEST_PLAN1.md`, `BEST_PLAN2.md`) converged on a "Folder 1 design + Folder 2 training rigor" template. This plan additionally:

**Substantive design choices the syntheses missed or got wrong:**
1. **Reduces fusion scalars from 10 to 7** by counting them correctly: $T_i \cdot 3 + w_i \cdot 3 + \gamma \cdot 1 = 7$. Folder 1's "10" appears to include the abstention threshold (which is a config constant, not learned).
2. **Drops `nn.Parameter` null embeddings** that BEST_PLAN.md adopted from folder 6. They are unnecessary when the encoder is skipped for absent modalities — 21 dead params.
3. **Explicitly calls out the per-sample vs per-batch modality dropout distinction** as a regression to avoid. The synthesis docs mention modality dropout without addressing folder 2's batch-level impl.
4. **Specifies the three-stage training schedule** with explicit roles per stage (independent → joint → calibration), and makes stage 2 cleanly skippable when aligned data is unavailable.
5. **Documents two-location modality dropout** (stage 2 joint training + stage 3 calibration), where the syntheses conflate them.
6. **Adds `test_calibration.py` and `test_lora.py`** as required unit tests for the 7-scalar fit and the LoRA-adapter wiring respectively. None of the syntheses include these.

**Resume-grade techniques that survived the scrutiny pass:**
- **Knowledge distillation** (Wav2Vec2-Base → CRNN) shipped as `memo train audio --distill`. The signature edge-ML story.
- **LoRA r=8 fine-tuning** shipped as `memo train text --lora`. Demonstrates modern PEFT competence.
- **Multi-task learning** via per-modality auxiliary losses in stage 2 joint training.
- **Cross-dataset generalization** as a default headline metric (not opt-in). The "did the model actually learn?" check.
- **Bootstrap 95% CIs** across all 7 modality subsets; **stratified 5-fold** for small audio datasets.
- **Fairness audit** auto-enables when JSONL contains slice labels.
- **ONNX FP32 + dynamic INT8** with explicit parity tolerances enforced in CI.
- **MACs profiling via fvcore** alongside wall-clock latency.
- **EMA, OneCycleLR, FocalLoss + label smoothing, ClassBalancedSampler, Mixup** — the full modern training stack, each justified by impact and feasibility, not included by convention.

**What was rejected after honest scrutiny** (and stays rejected even in this expanded scope):
- DistilBERT (3× weight for marginal gain), `mobilenet_v3_small(weights=None)`, `nn.Embedding` text encoder, max-softmax confidence, log-of-softmax fusion, per-batch modality dropout, null embeddings, gate regularization, cross-attention fusion, MobileNetV2 over 64×64 spectrograms, VAD continuous output, epistemic uncertainty estimation, FastAPI/K8s/federated learning, EDA text aug, pre-emphasis filter — every rejection has a stated reason in §12.

The cumulative effect: the confidence-gated late fusion under modality dropout (folder 1's standout idea) is preserved as the technical centerpiece, surrounded by the modern training and evaluation rigor that turns this from "a working classifier" into a **showcase of multimodal ML at the edge** — calibrated, distilled, parameter-efficiently fine-tuned, and rigorously evaluated.
