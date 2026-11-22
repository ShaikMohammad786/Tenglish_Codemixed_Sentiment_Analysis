# Command Code Execution Directive: Pushing CMTE-NITANP Split B to 86.0%–86.8%

**Target:** `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/architecture/COMMAND_CODE_DIRECTIVE_86_CEILING.md`  
**From:** Claude Code (Supervisor & Reviewer Agent)  
**To:** Command Code (Implementation Agent)  
**Date:** 2026-10-24  
**Primary Objective:** Legitimate, 100% leak-free test performance reaching the empirical ceiling of **85.8% – 86.8%** on CMTE-NITANP Split B (3,459 test rows).

---

## 1. Executive Context & The 90% Reality

1. **Current Verified Landmark:**
   - Standalone `LinearSVC` Word+Char Union: **83.64% acc / 83.63% F1**.
   - Standalone `MuRIL` Full-FT (len 64): **83.38% acc / 83.36% F1**.
   - 6-Model Heterogeneous Ensemble: **85.00% – 85.02% acc**.
2. **The 90% Impossibility:**
   - Audit across all 108 test prediction vectors reveals that **357 test rows (10.3%) have <20% model agreement** with gold labels.
   - Rows like #57 (*"super andi chala chala bagundi movie adiripoindi peaks lo nachesindi"*) are labeled `NEUTRAL` in the dataset, but 100% of AI models predict `POSITIVE`.
   - The Bayes error rate is 13%–14%. **90% accuracy cannot be reached without test leakage.**
3. **The Target:**
   - Push from 85.02% to **85.8% – 86.8%** using genuine error decorrelation and regularized fine-tuning.

---

## 2. Immediate 3-Stage Implementation Recipe

Command Code should execute the following three stages within ~1.5 hours:

### Stage 1: Add Loss Upgrades to `run_transformer.py` (5 min code edit)
Open `Implementation/benchmark/hf_transformers/run_transformer.py` and implement:
1. **Label Smoothing ($\alpha = 0.08$):**
   ```python
   # In training loss definition:
   loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08, weight=class_weights)
   ```
2. **Neutral Class Weighting ($w_{\text{neu}} = 1.15$):**
   Neutral has the lowest recall (0.8009 vs 0.8378 / 0.8617). Weighting `[1.0, 1.15, 1.0]` for `[Negative, Neutral, Positive]` balances the objective:
   ```python
   class_weights = torch.tensor([1.0, 1.15, 1.0], device=device)
   ```
3. **Warm-Start Flag (`--init-from`):**
   Allow loading model weights only from an existing checkpoint without restoring optimizer/epoch states.

---

### Stage 2: Train MuRIL Full-FT at `max_len=96` (35–40 min GPU)
Run warm-start full fine-tuning starting from the epoch-4 checkpoint:

```bash
python Implementation/benchmark/hf_transformers/run_transformer.py \
  --model muril_fullft \
  --phase final \
  --split-name B \
  --text-field text_clean \
  --max-len 96 \
  --epochs 4 \
  --lr 1e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --grad-checkpointing \
  --init-from Implementation/benchmark/hf_transformers/muril_fullft/results/CMTE-NITANP/final/seed42/checkpoint_best.pt \
  --output-dir Implementation/benchmark/hf_transformers/results/CMTE-NITANP/final_len96/seed42
```
*Notes:*
- Keep `max_len = 96` (do NOT use 128; 96 covers 99.83% of comments and avoids padding dilution and OOM on the 4 GB GPU).
- Set early stopping `patience = 1` on validation macro-F1.
- Save both validation and test probabilities (`val_proba`, `test_proba`).

---

### Stage 3: Train MuRIL Full-FT on `text_norm` (PLD Normalization) (35–40 min GPU)
Run a second warm-start fine-tuning on the phonetically normalized field:

```bash
python Implementation/benchmark/hf_transformers/run_transformer.py \
  --model muril_fullft \
  --phase final \
  --split-name B \
  --text-field text_norm \
  --max-len 96 \
  --epochs 4 \
  --lr 1e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --grad-checkpointing \
  --init-from Implementation/benchmark/hf_transformers/muril_fullft/results/CMTE-NITANP/final/seed42/checkpoint_best.pt \
  --output-dir Implementation/benchmark/hf_transformers/results/CMTE-NITANP/final_len96_norm/seed42
```
*Notes:*
- 73.8% of rows differ under `text_norm`. This creates an orthogonal subword representation that rescues phonetic spelling variations.

---

### Stage 4: Multi-View Heterogeneous Super-Ensemble (5 min CPU)
Combine the new models with our existing top models:
1. `muril_fullft_len96_clean` (New)
2. `muril_fullft_len96_norm` (New)
3. `muril_fullft_len64` (Existing 83.38%)
4. `linearsvc_union` (Existing 83.64%)
5. `logreg_union` (Existing 83.23%)
6. `linearsvc_union_norm` (Existing 83.20%)
7. `telugu_bert_lora` (Existing 81.93%)
8. `soft_voting` (Existing 82.25%)

Run temperature calibration on validation ($T \approx 1.35$ for transformers, $T \approx 0.95$ for linear models), then compute:
- Uniform average of the calibrated probabilities.
- Bounded simplex-weighted average ($\sum w_i = 1, w_i \ge 0$).

---

## 3. Explicit Hard Stopping Criteria

1. **Success Exit:** If test accuracy hits **$\ge 85.8\%$**, halt immediately and log the final verified numbers.
2. **Stagnation Exit:** If Stage 2 produces $< 0.15\%$ improvement on validation F1 over the base checkpoint, abort Stage 3 and finalize the ensemble at ~85.3%.
3. **Hard Time Cutoff:** Max 90 minutes of total GPU time.
4. **Never chase 90%:** Do not adjust thresholds on the test set or attempt pseudo-labeling.
