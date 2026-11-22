# FINAL PROJECT PLAYBOOK — Telugu Code-Mixed Sentiment Analysis
## Script-Invariant, De-biased, Fairness-Audited Sentiment (SID-CM) — A → Z, step by step

**Date:** 28 Sep 2026
**Compute:** free Kaggle/Colab T4 (16 GB) for training; laptop (RTX 2050 4 GB) for CPU work + inference
**Language:** **Telugu-English code-mixed** text
**Deliverables:** working code, fine-tuned model (Hugging Face Hub), and a paper targeting DravidianLangTech / IEEE ICAAIC / IEEE Access

> **How to read this:** start at §0, then follow §6 (phases) in order. Everything is pointwise. The math lives in §5.

---

## 0. DO THIS FIRST (the 3 highest-value actions)

1. **Build the leakage-controlled split** (de-duplicate + group by near-duplicate cluster) — your data foundation. *(~1 hour, CPU.)*
2. **Run a cheap TF-IDF + SVM baseline on the random split vs your clean split** — one number for "how much the split inflates." *(~15 min, CPU. One sentence in the paper.)*
3. **Train the first SID-CM model on Telugu** on Kaggle/Colab. *(~1–3 h GPU.)*

Nothing else matters until these three are done.

---

## 1. Final scope & research questions

**Task:** 3-class sentiment (Negative / Neutral / Positive) for **Telugu-English code-mixed** social-media text, with a clean, leakage-free evaluation.

- **RQ1:** Does training for script-invariance improve robustness when the text is written in Telugu script instead of Roman?
- **RQ2:** Does training for identity-invariance make the model **fairer** (smaller answer change under identity swaps)?
- **RQ3:** Can we give **calibrated** predictions and **abstain** when unsure?
- **RQ4:** Does the model generalize better on a leakage-controlled split than baselines?
- **RQ5 (bonus):** Does it transfer with few shots to a low-resource Dravidian language (e.g. Tulu)?

---

## 2. Novel contributions (final set — honest)

| # | Contribution | Prior art that exists | What is new |
|---|---|---|---|
| **C1** | **Leakage-controlled split** (0% near-duplicate overlap) | General contamination literature | **First for Telugu CMSA** *(config + one reported number)* |
| **C2** | **Adversarial script de-biasing** (trained-in) | ADAN (TACL 2018) = *language* invariance | **Script** (Roman↔Telugu) as the nuisance factor, for code-mixed sentiment |
| **C3** | **Cross-script consistency + script-TTA** | TransliCo (ACL 2024) pretraining alignment | Consistency + **inference-time script ensembling** for CMSA |
| **C4** | **Calibration + abstention** | Selective prediction in VLM/medical | **First** ECE + abstention report for Telugu CMSA |
| **C5** | **Adversarial identity de-biasing + counterfactual consistency** | INDIC-BIAS/IndiBias *audit* LLMs (English/Hindi) | **First model *trained* for identity-invariance** in Telugu sentiment |
| **C6** | **Cross-lingual few-shot transfer** (Telugu ↔ low-resource Dravidian) | Tulu transfer case studies | Family-scale shared encoder + few-shot, leakage-controlled |

**One-line pitch:** *Telugu sentiment models read the script instead of the meaning and may treat identity groups unequally — we **train one model to be invariant to both**, then calibrate it and let it abstain.*

---

## 3. Datasets (Telugu-English)

| Dataset | Content | Size | Labels | Source |
|---|---|---|---|---|
| **CMTET** (Kusampudi et al., RANLP 2021) — **primary** | Telugu-English social media | **19,857** sentences + word-level LID | 3-class (pos/neu/neg) | ACL Anthology / GitHub (`NsSuraj-01`) |
| IEEE-2025 Telugu study | YouTube + tweets | **19,868** comments | 3-class | IEEE 11346359 |
| `SaiMeghanath/tenglish-codemixed-nlp` | Telugu-En YouTube **news** comments | moderate | sentiment + toxicity + stance | GitHub |
| `mounikaiiith/Telugu_Sentiment` | Telugu | moderate | 3-class | Hugging Face |
| `Sumanth197/Code-Mixed-Text` | Telugu-En tweets | **5K** | 3-class | GitHub |
| *Optional (transfer)* | Tamil–Tulu corpus | Tamil 33,989+3,786; Tulu 6,458+781 | 4-class | ACL Anthology |

**Labels:** CMTET is already **3-class** → the core task needs **no harmonisation**. If you add a dataset with a different scheme, keep the extras as **auxiliary heads** (masked multi-task loss: `L = Σ_h mask_h(i)·L_h`).

**Grouping key:** CMTET likely has **no video IDs** → group the leakage-controlled split by **near-duplicate cluster** (MinHash).

---

## 4. Resources & tools

**Python libraries:** `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `scikit-learn`, `sentencepiece`, `datasketch` (MinHash), `pandas`, `numpy`, `scipy`, `matplotlib`, `faiss-cpu`.

**Transliteration / LID:** AI4Bharat **IndicXlit** (Roman↔**Telugu** script), **IndicLID** (word-level language ID; Telugu supported).

**Base models (one primary + one secondary):** `xlm-roberta-base` (278M) · `google/muril-base-cased` (236M) · `ai4bharat/indic-bert` (278M) · fallback `distilbert-base-multilingual-cased` (135M).

**Hardware:** Kaggle (2×T4, ~30 GPU-h/week) or Colab free (T4); laptop for data/eval.
**Storage:** Hugging Face Hub (public) for the model; GitHub for code.

---

## 5. Architecture & mathematics — SID-CM

### 5.1 Notation
`x` = Telugu-English comment · `x^R` = Roman view · `x^N` = Telugu-script view (IndicXlit) · `x^I` = identity-swapped counterfactual · `E` = encoder → `h ∈ ℝ^d` · `S` = sentiment head (K=3) · `A_s` = script discriminator · `A_i` = identity discriminator · `P` = projection head → `z` · `π_k` = class prior · `GRL_λ` = gradient-reversal layer.

### 5.2 Diagram
```
                    raw comment  x
        ┌───────────────┼────────────────────┐
        ▼               ▼                    ▼
   x^R (Roman)   x^N (Telugu script)      x^I (identity-swapped)
        └───────┬───────┘                    │
                ▼                            │
        shared encoder E  (XLM-R-base / MuRIL + LoRA)
                │  h                         │
   ┌────────────┼──────────────┬─────────────┼──────────────┐
   ▼            ▼              ▼             ▼              ▼
sentiment   script-adv    identity-adv   projection   (fairness uses S(x^I))
 head S     head A_s      head A_i        head P
   │        (GRL)         (GRL)            │
   │            │              │           │
   │      L_script_adv    L_id_adv     L_contrast (x^R ↔ x^N)
   │
   ├── L_script_consist   = KL( S(x^R) ‖ S(x^N) )
   └── L_fairness_consist = KL( S(x)   ‖ S(x^I) )

 INFERENCE:  p̄ = ½(S(x^R)+S(x^N)) → temperature scaling → calibrated p → abstain if max p < θ
```

### 5.3 Losses (mathematical form)
**(1) Class-balanced focal loss + logit adjustment**
```
ℓ'_k = ℓ_k + τ·log π_k ;  p = softmax(ℓ')
L_sent = − Σ_k α_k (1 − p_k)^γ · y_k · log p_k        (α_k ∝ 1/π_k)
```
**(2) Adversarial script removal (GRL)**
```
L_script_adv = − Σ_{s∈{R,N}} 1[script(x)=s] · log A_s(GRL_λ(h))_s
GRL_λ : forward = identity ; backward: ∂L/∂h ← −λ·∂L/∂h
```
**(3) Adversarial identity removal (GRL)**
```
L_id_adv = − Σ_g 1[identity(x)=g] · log A_i(GRL_λ(h))_g
```
**(4) Transliteration contrastive (InfoNCE)**
```
L_contrast = − log [ exp(sim(z^R, z^N)/τ_c) / Σ_j exp(sim(z^R, z_j)/τ_c) ]
```
**(5) Script consistency**
```
L_script_consist = KL( S(x^R) ‖ S(x^N) )
```
**(6) Fairness consistency (counterfactual)**
```
L_fairness_consist = KL( S(x) ‖ S(x^I) )
```
### 5.4 Total objective
```
L = L_sent + λ₁·L_script_adv + λ₂·L_id_adv + λ₃·L_contrast + λ₄·L_script_consist + λ₅·L_fairness_consist
```
Adversarial weights `λ₁, λ₂` **ramped 0 → 1** over the first ~30% of steps. Defaults: `γ=2`, `τ=1`, `τ_c=0.07`, `λ₃=λ₄=λ₅=1`.

### 5.5 Inference
```
p̄ = ½(S(x^R)+S(x^N)) ;  T = fit_temperature(p̄, dev) ;  p_cal = softmax(log p̄ / T)
ŷ = argmax p_cal if max p_cal ≥ θ else ABSTAIN
```

### 5.6 Contribution → metric map (every claim is measured)
| Mechanism / step | Fixes | Metric |
|---|---|---|
| `L_sent` (focal + logit adj.) | class imbalance | macro-F1, minority F1 |
| `L_script_adv` + `L_contrast` + `L_script_consist` | script reliance | **Δ_script** |
| `L_id_adv` + `L_fairness_consist` | identity reliance | **identity-flip rate** |
| script-TTA | script noise | Δ_script, macro-F1 |
| temperature scaling | overconfidence | **ECE** |
| abstention θ | unknown unknowns | **risk–coverage** |
| leakage-controlled split (C1) | eval validity | **F1 on Split B** |
| retrieval/few-shot (C6) | rare classes / transfer | minority F1, **transfer F1** |

---

## 6. STEP-BY-STEP PIPELINE

### Phase 0 — Environment (once)
1. Create a Python 3.10 env; `pip install` the libraries in §4.
2. Make folders: `data/`, `scripts/`, `models/`, `results/`, `paper/`.
3. Create a Kaggle account; verify GPU access. *(CPU-only for Phases 1–5.)*

### Phase 1 — Download data (CPU)
4. Download **CMTET** (Telugu-English sentiment, ~19.9k). Optionally add a backup Telugu set.
5. Record rows, label counts, and whether any **video/source column** exists.

### Phase 2 — Preprocess (CPU, identical across splits)
6. Minimal cleaning: lowercase, strip URLs/@mentions, collapse repeated chars. **Keep emojis/punctuation**; keep a raw copy.
7. Save `clean.csv` + `raw.csv`.

### Phase 3 — Label check (CPU)
8. Confirm CMTET labels map to core 3-class (pos/neu/neg); keep any extras as auxiliary heads.
9. Save `labeled.csv`: `text, sentiment, lang, script`.

### Phase 4 — De-duplicate + build the leakage-controlled split (CPU)
10. Remove exact duplicates (normalised text).
11. Near-duplicate detection: **MinHash + LSH** (`datasketch`), Jaccard ≥ 0.8 → **cluster IDs**.
12. Build split **A (random 90/5/5)** and split **B (leakage-controlled, grouped by cluster, 0% overlap)**. *(Optional split C: temporal if dates exist.)*
13. **Cheap leakage number:** TF-IDF+SVM on A and B → `macro-F1(A) − macro-F1(B)`.

### Phase 5 — Views + counterfactuals (CPU)
14. Generate `x^N` (Roman→Telugu script) with **IndicXlit** for every item.
15. Build a **Telugu identity lexicon** (caste/religion/region/gender: Telugu + Roman forms; e.g. Andhra/Telangana regions).
16. Generate `x^I` (identity-swapped) pairs for comments containing identity terms.
17. Store `x, x^N, x^I` side by side (token-aligned).

### Phase 6 — Train SID-CM (GPU)
18. Tokenize all views (`max_len=96`); build transliteration contrastive pairs.
19. Config: XLM-R-base/MuRIL + **LoRA r=16 α=32**; lr 2e-4; batch 16; 3 epochs; **3 seeds**.
20. Train with the **full 6-term objective** (§5.4); ramp `λ₁, λ₂` 0→1 over 30% of steps.
21. Checkpoint `save_steps=250` → Drive/HF Hub; resume with `resume_from_checkpoint=True`.
22. Train on **Split B** (main); one control run on **Split A**.

### Phase 7 — Evaluate (CPU)
23. Report **macro-F1 (primary)**, weighted-F1, accuracy, **per-class P/R/F1**, **MCC**, bootstrap 95% CIs.
24. Compare against Telugu baselines on the *same* splits.

### Phase 8 — Robustness + calibration + abstention (CPU)
25. **Δ_script:** transliterate the test set → re-run → F1 drop; also the **script-TTA** gain.
26. **Calibration:** temperature scaling on dev → **ECE** + reliability diagram.
27. **Abstention:** sweep θ → **risk–coverage curve**.

### Phase 9 — Fairness audit (verification of the trained-in fairness) (CPU)
28. Build a **counterfactual probe set** (identity swaps, sentiment fixed).
29. Measure the **flip rate** per identity group; compare SID-CM vs baselines.

### Phase 10 — Retrieval memory (C6, optional) (CPU)
30. FAISS index of labelled train comments; fuse `p_final = λ·p_model + (1−λ)·p_knn`; re-measure rare-class F1.

### Phase 11 — Compare, write, release
31. Fill the comparison table (§8). State clearly what you win.
32. Release code + checkpoint + probe set + splits.

---

## 7. Training time (compute)

| Job | Free T4 | RTX 2050 4 GB (QLoRA) |
|---|---|---|
| One run, **Telugu (~19.9k, 3 epochs)** | ~1–3 h | ~6–12 h |
| Main experiment (2 splits × 3 seeds) | ~8–18 h | too slow → T4 |
| Ablations (~8 variants) | ~10–20 h | — |
| Eval + calibration + fairness audit | < 1 h (CPU) | CPU |
| **Total** | **~20–45 GPU-h** | — |

→ **Fits Kaggle's free 30 GPU-h/week.** Checkpoint + resume so resets cost ~1 min.

---

## 8. Comparison targets (Telugu — no shared-task leaderboard)

| Target | Reported | How we compare |
|---|---|---|
| **CMTET baseline** (Kusampudi et al., RANLP 2021) | **~80.2% accuracy** (unsup. normalization + MLP) | same split |
| IEEE 2025 Telugu study | 19,868 comments; transformer baselines | same split |
| Other Telugu code-mixed sentiment papers | varied (non-standard splits) | reproduced where possible |

**Baselines to reproduce:** majority · TF-IDF+SVM/LR · fastText · BiLSTM · CNN+BiLSTM · mBERT · XLM-R-base · IndicBERT · MuRIL · plain LoRA (no SID-CM).

**Win:** Δ_script · ECE · abstention · **identity-flip rate** · Split-B generalization · transfer F1.
**Tie:** raw accuracy/macro-F1 on the random split. **We do not claim SOTA.**

---

## 9. Ablations (prove each part matters)

A1 no script-adversarial head · A2 no identity-adversarial head · A3 no transliteration contrastive · A4 no script-consistency · A5 no fairness-consistency · A6 no script-TTA · A7 no calibration · A8 full fine-tune vs LoRA · A9 Roman-only (single view) · A10 with/without retrieval memory.

---

## 10. Publication plan

- **Venue 1 (fast):** DravidianLangTech workshop (ACL/NAACL co-located).
- **Venue 2:** IEEE ICAAIC / IEEE conference (code-mixed sentiment appears here).
- **Venue 3:** IEEE Access (APC) or *Social Network Analysis & Mining*.

**Honest positioning:** a **method + fairness-audit** paper for an under-served language (Telugu) — novel in combination, feasible on free hardware, reproducible.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| No video IDs → grouping weak | Group by **near-duplicate cluster** (still 0% near-dup overlap) |
| Adversarial heads hurt F1 | Ramp λ₁, λ₂ slowly; keep small; ablate |
| Contrastive needs large batches | Memory queue (MoCo-style) |
| Telugu identity lexicon incomplete | Start with a small verified list; report coverage |
| Transliteration errors | Ensemble transliterators; report sensitivity |
| Only 1 local overnight run | Train on Kaggle/Colab; checkpoint to Drive/HF Hub |

---

## 12. Reproducibility checklist
Pin model revisions · fix seeds · release splits + probe set + code · report CIs and significance · publish the checkpoint on HF Hub · document the label-mapping table · include a datasheet.

---

## 13. What NOT to do
- Don't claim SOTA F1.
- Don't train XLM-R-large, MoE, or do multi-day jobs.
- Don't build a new annotated corpus (out of scope).
- Don't hand-edit the taste files; use the `taste` tool.
- Don't skip checkpoints — sessions reset.

---

## 14. Reference anchors (verified)
CMTET / Kusampudi et al. (RANLP 2021) · IEEE-2025 Telugu CM sentiment study · ADAN (TACL 2018) · TransliCo (ACL 2024) · "Multilingual LMs Encode Script Over Linguistic Structure" (ACL 2026) · SenTAS++ (IEEE 2025) · Share What You Already Know (ACM TALLIP 2024) · MultiSent-RAG (IP&M 2026) · INDIC-BIAS (ACL 2025) / IndiBias (NAACL 2024) · dedup-vs-memorization (arXiv 2508.14062).

---

### The A→Z in one line
**Download CMTET → clean → check 3-class labels → dedupe + build the leakage-free split (+ cheap TF-IDF leakage number) → make Telugu-script views + identity-swapped counterfactuals → train SID-CM (script-adversarial + identity-adversarial + transliteration-contrastive + two consistency losses, LoRA) on a free T4 → evaluate F1/Δ_script/ECE/abstention/fairness → compare to Telugu baselines → write it up for DravidianLangTech/IEEE.**
