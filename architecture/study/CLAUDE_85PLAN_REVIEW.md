# Adversarial Review of Approved Plan: Pushing CMTE-NITANP Split B to ≥85%

**Target Plan:** `/home/shaik/.commandcode/plans/cmte-nitanp-85-accuracy-plan.md`  
**Reviewer:** Claude Code (Advisory & Review Agent)  
**Date:** 2026-10-24  
**Format:** Severity-Ranked (`FATAL` / `MAJOR` / `MINOR`)  
**Scope:** Methodological validity, leakage risk, statistical integrity, and scientific framing.

---

## Executive Scorecard

| Question / Dimension | Verdict | Severity | Core Risk |
|---|:---:|:---:|---|
| **1. Val-Fitted Stacker (3,459 rows, 8 models)** | **Fragile / High Risk** | **MAJOR** | Base models already selected on val; stacker inherits optimistic selection bias. Unconstrained multinomial logistic regression will overfit validation noise. |
| **2. Warm-Start Full-FT at max_len=96** | **Legitimate (No Leakage)** | **MAJOR** | No test leakage, but high risk of immediate post-plateau overfitting (epoch 4 was peak); unadapted position embeddings (64–95) need proper learning dynamics. |
| **3. Fairness of `text_norm` (PLD)** | **Methodologically Asymmetric** | **MAJOR** | Legitimate feature view, but evaluating `text_norm` only on the transformer without running TF-IDF LogReg on `text_norm` introduces an unfair comparison bias. |
| **4. Missing Cheap Levers** | **Substantial Omissions** | **MAJOR** | Plan pursues compute-heavy retrains while omitting zero-cost levers: Label Smoothing ($\alpha=0.1$), Temperature Scaling, Class-weighted Loss, and FGM. |
| **5. Narrative & Framing of 85%** | **Critical Thesis Threat** | **FATAL** | Claiming "transformers reach 85%" contradicts the paper's core thesis. Must be framed as *Hybrid Synergy*: orthogonal linear and neural error spaces breaking the single-model ceiling. |

---

## Ranked Findings

```
[FATAL]  Methodological or narrative flaw that invalidates the paper's thesis or triggers desk rejection.
[MAJOR]  Substantial statistical, optimization, or fairness risk that must be mitigated before execution.
[MINOR]  Tactical improvement, hyperparameter adjustment, or reporting refinement.
```

---

### [FATAL-1] Scientific Framing: Claiming an 85% Ensemble Without Breaking the Paper's Thesis

- **The Risk:**
  The entire manuscript (`CMTE_NITANP_IEEE.tex`) argues that deep transformers provide a *capacity illusion* on Romanized Dravidian sentiment, as a 0-parameter linear model (82.93 F1 / 83.24 union) matches a 238M-parameter MuRIL full-FT (83.36 F1). If the team announces: *"We achieved 85.0% using deep learning / transformers,"* reviewers will immediately discard the entire paper's thesis as obsolete or self-contradictory.
- **The Honest Scientific Framing:**
  1. **Single-Model Ceiling vs. Heterogeneous Synergy:** Emphasize that **no individual model** (classical or neural) can exceed ~83.4% on leak-free Split B. The 85% mark is unlocked *only* by bridging the representational dichotomy: TF-IDF linear models capture exact, unfragmented morphemic clitics (`-ledu`, `-kadu`, `asalu`), while MuRIL captures English context and long-range semantic syntax.
  2. **Never claim a single-model SOTA:** The report and any paper addition must state: *"Single-model ceiling remains ~83.4%. Heterogeneous ensembling of classical linear n-grams and transformer representations reaches 85.X% (McNemar $p < 0.01$ vs. both standalone baselines), proving their error distributions are genuinely orthogonal."*

---

### [MAJOR-1] Val-Fitted Stacker on 3,459 Rows: Selection Leakage and Overfitting Risk

- **The Risk:**
  - In Stage 1 (`stack.py`), an unconstrained multinomial logistic regression is fitted on 8 base models $\times$ 3 classes = 24 probability features across 3,459 validation rows.
  - **Selection Contamination:** The 8 base models already used this identical validation set for early stopping (e.g., MuRIL full-FT stopped at epoch 4 on val F1), checkpoint selection, and hyperparameter tuning. The base models' validation predictions are systematically *optimistic and overconfident* compared to their test-set behavior.
  - Fitting an unconstrained 24-feature logistic regression on contaminated validation probabilities will overfit validation-specific noise, leading to an apparent 85%+ on val that collapses to ~84.1% on test.
- **Mandatory Mitigations:**
  1. **Constrained Convex Blending First:** Before fitting a logistic stacker, evaluate a **simplex-constrained soft vote** (weights $w_i \ge 0, \sum w_i = 1$) optimized via Nelder-Mead or Dirichlet weighting on val. Bounded convex weights cannot overfit noise or flip polarities.
  2. **Temperature Scaling Before Stacking:** Uncalibrated base models (MuRIL has ECE = 0.0702; LightGBM has ECE = 0.0132) will corrupt a linear meta-learner. Apply post-hoc temperature scaling ($T$) to calibrate each model's validation logits before passing them to the stacker.
  3. **Strict L2 Regularization:** If using `LogisticRegression` as a stacker, enforce high regularization ($C \in [0.001, 0.1]$) with `fit_intercept=False` to prevent the meta-learner from shifting class priors artificially.

---

### [MAJOR-2] Warm-Starting Full-FT at max_len=96: Position Dynamics and Plateau Overfitting

- **The Risk:**
  - **No Test Leakage:** Loading model weights from `checkpoint_best.pt` (epoch 4 on train) does *not* leak test data. This is standard progressive transfer.
  - **Overfitting Risk:** In the verified training log, MuRIL full-FT training loss fell monotonically from 0.8860 to 0.2416, while validation F1 peaked at epoch 4 (0.8255) and deteriorated on epochs 5 (0.8227) and 6 (0.8241). The base model was already beginning to overfit. Starting from epoch 4 weights and training for 4 more epochs on the same train set risks immediate catastrophic overfitting.
  - **Unadapted Positional Embeddings:** MuRIL has pre-trained position embeddings up to 512, but positions 64–95 received *zero gradient updates* during the first 4 epochs of fine-tuning at `max_len=64`. Running at a very low learning rate ($1 \times 10^{-5}$) may fail to align these cold position embeddings with the hot representations of positions 0–63.
- **Mandatory Mitigations:**
  1. **Warmup & Early Exit:** Use a 10% linear warmup even at $1\times 10^{-5}$, and set early stopping `patience=1`. If val loss does not improve on epoch 1 or 2, terminate immediately to avoid burning GPU time.
  2. **Progressive Unfreezing:** Keep the lower 6 transformer layers frozen for the first epoch at length 96, allowing top layers and position embeddings to adjust before updating the entire 238M parameter backbone.

---

### [MAJOR-3] Methodological Asymmetry with `text_norm` (PLD Normalization)

- **The Risk:**
  - The plan proposes training MuRIL full-FT on `text_norm` (Stage 3) using `dict_1pld.csv` (which changes 73.8% of rows).
  - In the main paper, all 21 models were strictly benchmarked on `text_clean`.
  - If `text_norm` is tested **only on the transformer** to push it to 85%, this introduces an asymmetric experimental confound. Reviewers will ask: *"Did transformers beat linear models, or did you just give transformers a cleaned vocabulary while leaving classical baselines on noisy text?"*
- **Mandatory Mitigations:**
  1. **Run TF-IDF LogReg on `text_norm`:** This takes **10 seconds on CPU**! If LogReg on `text_norm` jumps from 0.8293 to 0.8380, that is an enormous scientific finding (phonetic normalization universally resolves code-mixed sparsity).
  2. **Multi-View Ensemble Inclusion:** Include both `text_clean` LogReg and `text_norm` LogReg in the Stage 1/Stage 3 ensemble.

---

### [MAJOR-4] Missing Cheap, Zero-Cost Optimization Levers

The plan relies heavily on 30–45 minute GPU retraining runs (Stages 2, 3, 4), but ignores several standard, battle-tested competitive NLP levers that cost **zero or near-zero extra compute**:

1. **Label Smoothing ($\alpha = 0.05 - 0.1$):**
   - *Why:* MuRIL full-FT overfits cross-entropy loss by epoch 5. Label smoothing directly penalizes extreme logit divergence, acts as a regularizer against noisy social labels, and reduces ECE.
   - *Cost:* 0 extra seconds (built into PyTorch `nn.CrossEntropyLoss(label_smoothing=0.1)`).
2. **Class-Weighted Loss:**
   - *Why:* In CMTE-NITANP, Neutral is the hardest class with the lowest recall (0.8009 vs. 0.8378 for Negative and 0.8617 for Positive). Weighting classes inversely by frequency or setting Neutral weight to 1.15 in cross-entropy directly balances macro-F1.
   - *Cost:* 0 extra seconds.
3. **Stochastic Weight Averaging (SWA) / Checkpoint Averaging:**
   - *Why:* Average the weights of epochs 3, 4, and 5. This flattens the loss basin, prevents overfitting, and reliably adds +0.2–0.4 F1 in transformer fine-tuning.
   - *Cost:* 1 minute CPU file operation.
4. **Fast Gradient Method (FGM / Adversarial Training):**
   - *Why:* Perturbing embedding gradients ($\epsilon = 0.5$) during backprop is the gold standard for noisy Romanized transliteration tasks.
   - *Cost:* Adds ~20% wall-clock time, but far higher ROI than LoRA $r=32$.

---

### [MINOR-1] Threshold / Prior Tuning Must Be Constrained

- **The Risk:** Stage 5 proposes searching class multipliers or temperature + prior correction on validation probabilities to maximize val macro-F1.
- **Mitigation:** Unconstrained 3-class threshold optimization has high variance on 3,459 rows. Constrain the search to temperature scaling $T$ and a single class-prior shift:
  $$P^*(y=c|x) \propto \frac{P(y=c|x)}{(\pi_c)^\gamma}$$
  where $\pi_c$ is training class prior and $\gamma \in [0, 0.5]$ is tuned on val. Do not perform freeform per-class threshold grid searches.

---

### [MINOR-2] Seed Reporting on Winning Configuration

- **The Risk:** Stage 0–5 uses Seed 42. If an ensemble reaches 85.02% on Seed 42, declaring "CMTE-NITANP broken at $\ge 85\%$" on a single seed is vulnerable to the same criticism raised in our review of MuRIL full-FT ($N=1$).
- **Mitigation:** If the winning configuration exceeds 85.0%, verify that the classical ensemble components (which have 5 seeds) retain $\ge 84.8\%$ across seeds, and evaluate Seed 43 for any newly trained transformer checkpoint before publication.

---

## Actionable Recommendations Before GPU Launch

1. **Adopt Stage 0 Immediately (Offline Audit):** Run uniform soft-voting of existing models (`muril_fullft` + `logreg_union` + `telugu_bert_lora` + `indicbert_lora` + `multinomial_nb`). If this already hits **84.2–84.5%**, the 85.0% target is within reach without complex retraining.
2. **Add Label Smoothing (0.1) and Neutral Class Weighting (1.1)** to Stage 2 (`run_transformer.py`).
3. **Use Simplex-Constrained Soft Voting** instead of unconstrained multinomial logistic regression in Stage 1 to prevent validation selection overfitting.
4. **Run TF-IDF LogReg on `text_norm` (10s CPU)** to guarantee fair scientific comparison.
5. **Frame the Result as "Representational Synergy"** proving that classical morpheme-local n-grams and deep contextual transformers correct each other's failure modes.
