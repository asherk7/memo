# Math and ML

The equations behind the fusion layer, the training losses, the calibration
objective, and the evaluation metrics. The system-design view is in
[architecture.md](architecture.md).

Throughout, there are $K = 7$ classes. Encoder $i \in \{\text{image, text, audio}\}$
produces raw logits $z_i \in \mathbb{R}^K$.

## Confidence-gated late fusion

The fusion layer has seven trainable scalars: a temperature $T_i$ and a weight
$w_i$ per modality, and one sharpness $\gamma$. The abstention threshold $\tau$ is a
fixed config constant, not learned.

**1. Temperature-scaled distributions.** Each modality's logits become a
probability distribution, scaled by its own temperature:

$$p_i = \mathrm{softmax}(z_i / T_i) \in \Delta^{K-1}.$$

Temperatures are stored in log-space ($T_i = e^{\theta_i}$) so they stay strictly
positive under gradient updates.

**2. Confidence via normalized inverse entropy.** Each distribution gets a
confidence score in $[0, 1]$:

$$c_i = 1 - \frac{H(p_i)}{\log K}, \qquad H(p_i) = -\sum_{k=1}^{K} p_{i,k}\,\log p_{i,k}.$$

A one-hot (certain) distribution has $H = 0 \Rightarrow c_i = 1$; a uniform
(uninformative) one has $H = \log K \Rightarrow c_i = 0$. Entropy is computed in
log-space and $c_i$ is clamped to $[0,1]$ for numerical safety.

**3. Gating weights.** Each present modality gets an unnormalized weight combining
its learned weight, a presence mask $m_i \in \{0,1\}$, and its confidence raised to
the sharpness:

$$\alpha_i = \mathrm{softmax}(w)_i \cdot m_i \cdot c_i^{\gamma}.$$

$\gamma = 0$ recovers a plain learned weighted average; $\gamma > 0$ progressively
down-weights uncertain modalities.

**4. Renormalize and fuse.** Weights renormalize over the present modalities, and
the fused distribution is their weighted average:

$$\tilde{\alpha}_i = \frac{\alpha_i}{\sum_j \alpha_j}, \qquad
p_{\text{final}} = \sum_i \tilde{\alpha}_i\, p_i, \qquad
\hat{y} = \arg\max_k\, p_{\text{final},k}.$$

The denominator is floored at $10^{-12}$ to avoid a $0/0$ when every present
modality is maximally uncertain and $\gamma > 0$. Because the weights renormalize
over present modalities only, the output depends solely on the modalities that
actually contributed — masking is explicit, not inferred from zeroed logits.

**5. Abstention.** If $\max_k p_{\text{final},k} < \tau$, the prediction is flagged
`abstained`. The caller decides what to do with it.

## Training losses

**Focal loss with label smoothing.** A single combined loss (not focal stacked on
top of smoothing), with the focal modulation applied to a label-smoothed
cross-entropy:

$$\mathcal{L}_{\text{focal}} = (1 - p_y)^{\gamma_f}\;\Big(-\sum_{k} q_k \log p_k\Big),$$

where $p = \mathrm{softmax}(z)$, $p_y$ is the true-class probability, the focusing
parameter is $\gamma_f = 2.0$, and $q$ is the smoothed target with smoothing
$\epsilon = 0.05$:

$$q_k = (1 - \epsilon)\,\mathbb{1}[k = y] + \frac{\epsilon}{K}.$$

The focal term $(1 - p_y)^{\gamma_f}$ shrinks the loss on already-confident correct
examples so training focuses on hard ones. With $\gamma_f = 0$ and $\epsilon = 0$
this reduces to ordinary cross-entropy.

**Effective-number class weights** (Cui et al., 2019). To counter class imbalance,
each class $c$ is weighted by the inverse of its effective sample count:

$$\alpha_c = \frac{1 - \beta}{1 - \beta^{\,n_c}}, \qquad \beta = 0.9999,$$

normalized to mean 1, where $n_c$ is the class count. These weights multiply the
per-sample focal loss and complement a class-balanced sampler at the data layer.

**Knowledge distillation** (Hinton-style). The audio student is trained against a
blend of the hard focal loss and a soft KL term matching a frozen teacher:

$$\mathcal{L}_{\text{KD}} = \alpha\,\mathcal{L}_{\text{focal}}(\hat{y}_s, y)
+ (1 - \alpha)\,\tau_{\!d}^{2}\;
\mathrm{KL}\!\big(\sigma(\hat{y}_t / \tau_{\!d}) \,\|\, \sigma(\hat{y}_s / \tau_{\!d})\big),$$

with $\alpha = 0.5$ and distillation temperature $\tau_{\!d} = 4$. The $\tau_{\!d}^2$
factor keeps the soft-term gradient magnitude comparable to the hard term. At
$\alpha = 1$ this reduces to the plain focal loss.

## Calibration objective

The fusion scalars are fit by minimizing the negative log-likelihood of the fused
distribution on an aligned validation set, under per-sample modality dropout:

$$\min_{\{T_i, w_i, \gamma\}}\;
\mathbb{E}_{(x, y)}\;\mathbb{E}_{m \sim \text{Dropout}}\;
\big[-\log p_{\text{final}, y}(x; m)\big].$$

Each modality is masked independently per sample (Bernoulli, drop rate 0.3, text
0.15), so the seven scalars learn to perform across all $2^K{-}1 = 7$ modality
subsets simultaneously rather than only the all-present case. The encoders are
frozen, so their logits are constant and precomputed once; optimization (AdamW,
lr $10^{-2}$, ~200 epochs) then runs over the cache in seconds.

## Evaluation metrics

Predictions and labels over $N$ samples; $\mathrm{TP}_k, \mathrm{FP}_k, \mathrm{FN}_k$
are per-class counts from the confusion matrix.

**Per-class precision / recall / F1.**

$$P_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FP}_k}, \quad
R_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FN}_k}, \quad
F1_k = \frac{2 P_k R_k}{P_k + R_k}$$

(each defined as 0 where its denominator is 0).

**macro-F1** averages $F1_k$ over all $K$ classes (an absent class contributes 0).
**weighted-F1** averages $F1_k$ weighted by class support $n_k$:
$\sum_k (n_k / N)\,F1_k$.

**UAR** (unweighted average recall, the speech-emotion standard) averages recall
over the classes present in the slice:

$$\mathrm{UAR} = \frac{1}{|\mathcal{P}|}\sum_{k \in \mathcal{P}} R_k, \qquad
\mathcal{P} = \{k : n_k > 0\}.$$

**Expected Calibration Error (ECE, 15 bins).** Group predictions into $M = 15$
equal-width bins by confidence $\hat{p} = \max_k p_k$. With accuracy
$\mathrm{acc}(b)$ and mean confidence $\mathrm{conf}(b)$ in bin $b$:

$$\mathrm{ECE} = \sum_{b=1}^{M} \frac{|b|}{N}\,\big|\,\mathrm{acc}(b) - \mathrm{conf}(b)\,\big|.$$

**Brier score** (multiclass) is the mean squared error against the one-hot target
$y \in \{0,1\}^K$, in $[0, 2]$:

$$\mathrm{Brier} = \frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} \big(p_{n,k} - y_{n,k}\big)^2.$$

ECE and Brier probe different aspects of calibration — ECE measures bin-level
reliability, Brier is a strictly proper score over the whole simplex — so both are
reported.
