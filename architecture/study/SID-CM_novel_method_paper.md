# SID-CM: Script-Invariant, De-biased, Fairness-Audited Sentiment Analysis for Telugu-English Code-Mixed Text
## A complete, end-to-end research paper plan — math, architecture, implementation, training, contributions

**Date:** 28 Sep 2026 · **Compute target:** free Kaggle/Colab T4 (16 GB) or local RTX 2050 (4 GB, QLoRA) · **Focus:** **Telugu (+ English)** code-mixed text; optional cross-lingual transfer to a second Dravidian language
**Companion docs:** `literature_review_...md`, `FINAL_PROJECT_PLAYBOOK_SID-CM.md`, `FINAL_SAMVAD_research_blueprint.md`

> **Honesty rule:** every "novelty" is stated *with the prior art that already exists*, and exactly what remains new. Nothing here is claimed as an invention that isn't.

---

## 0. One-paragraph summary

We build one end-to-end pipeline — **download → preprocess → de-duplicate → split → train → calibrate → evaluate → compare** — on **Telugu-English code-mixed text**:
- **Novelty 1 (data/evaluation):** a **leakage-controlled split** with **0% near-duplicate overlap**, plus the first quantification of the leakage gap for Telugu CMSA.
- **Novelty 2 (method):** **SID-CM**, a model **trained to be invariant to two nuisance factors at once — script and identity** — while staying sensitive to sentiment. It combines **adversarial de-biasing** (gradient reversal), **transliteration contrastive learning**, and **two counterfactual consistency losses** (script + identity), then adds **multi-script test-time ensembling** and **calibrated abstention**.
Everything runs on a free T4.

---

## 1. The problem (why this is worth publishing — for Telugu)

1. **Telugu has no standard sentiment benchmark at all.** Telugu was **excluded** from DravidianCodeMix and from the recurring Dravidian sentiment shared task (it only appeared in hate-speech and prompt tasks). So even the evaluation protocol is unsettled → inflated/uncontrolled splits and no longitudinal comparison.
2. **Models read the script, not the meaning.** A 2026 ACL finding shows multilingual LMs **"encode script over linguistic structure"** — the same Telugu comment in Roman vs native script can flip a prediction.
3. **Models may treat identity groups unequally.** No one has checked whether Telugu sentiment models change their answer when caste/religion/region/gender words change.
4. **No reliability reporting.** No Telugu (or Dravidian) code-mixed sentiment paper reports **calibration (ECE)** or **abstention**.

**Core idea — train the model to be blind to *how* it's written and *who* it's about, but sensitive to *what* is said.**

---

## 2. Novel contributions (honest: prior art + what is new)

| # | Contribution | Closest prior art (exists) | What is genuinely **new** | Feasible on T4? |
|---|---|---|---|---|
| **C1** | **Leakage-controlled split + leakage-gap quantification** | General contamination literature; dedup papers | **First for Telugu CMSA** — 0% train/test near-dup overlap, grouped/temporal split, measured Δ | ✅ (CPU) |
| **C2** | **Adversarial *script* de-biasing** | **ADAN (TACL 2018)** did adversarial *language* invariance for sentiment | **Script** (Roman vs native) as the nuisance factor, for code-mixed sentiment | ✅ |
| **C3** | **Cross-script consistency + multi-script inference ensembling (script-TTA)** | TransliCo (ACL 2024) aligns scripts during *pretraining* | Consistency + **inference-time ensembling over script variants** for CMSA | ✅ |
| **C4** | **Calibrated abstention (selective prediction)** | Selective prediction exists in VLM/medical | **First** ECE + abstention analysis for Telugu CMSA | ✅ |
| **C5** | **Fairness: adversarial identity de-biasing + counterfactual consistency** | INDIC-BIAS / IndiBias audit LLMs (English/Hindi) | **First model *trained for* identity-invariance in Telugu sentiment** | ✅ |
| **C6** | **Cross-lingual few-shot transfer (Telugu ↔ a low-resource Dravidian language, e.g. Tulu)** | case studies exist for Tulu transfer | Family-scale shared encoder + few-shot, under the leakage-controlled protocol | ✅ |

> **Positioning:** *We do not invent adversarial invariance, contrastive learning, or script alignment. We are the first to (i) give Telugu code-mixed sentiment a leakage-controlled evaluation, and (ii) make one model simultaneously invariant to script and identity, with calibrated abstention.*

---

## 3. Architecture & mathematics — SID-CM (full specification)

### 3.1 Notation
| Symbol | Meaning |
|---|---|
| `x` | raw code-mixed comment (Telugu-English) |
| `x^R` | Roman view of `x` |
| `x^N` | native-script view (Telugu script, via IndicXlit) |
| `x^I` | identity-swapped counterfactual of `x` |
| `E(·)` | shared multilingual encoder → `h ∈ ℝ^d` |
| `S(·)` | sentiment head (K = 3 classes) |
| `A_s(·)` | script discriminator (Roman/Telugu) |
| `A_i(·)` | identity discriminator (identity group / present-absent) |
| `P(·)` | projection head → unit-sphere vector `z` |
| `π_k` | class prior of class k (training frequency) |
| `γ, τ, τ_c` | focal exponent, logit-adjustment temperature, InfoNCE temperature |
| `GRL_λ` | gradient-reversal layer with weight λ |

### 3.2 Diagram
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

### 3.3 Heads
```
h        = E(x)
ℓ        = W_s h + b_s                 # sentiment logits
script   = A_s( GRL_λ(h) )             # adversarial
identity = A_i( GRL_λ(h) )             # adversarial
z        = normalize( P(h) )           # for contrastive
```

### 3.4 Losses (mathematical form)

**(1) Class-balanced focal loss + logit adjustment** (fixes class skew):
```
ℓ'_k   = ℓ_k + τ · log π_k                       # prior correction
p      = softmax(ℓ')
L_sent = − Σ_k  α_k (1 − p_k)^γ · y_k · log p_k
```
`α_k ∝ 1/π_k` (class weights); `y` = one-hot label.

**(2) Adversarial script removal** (gradient reversal):
```
L_script_adv = − Σ_{s∈{R,N}}  1[script(x)=s] · log A_s(GRL_λ(h))_s
GRL_λ : forward = identity ;  backward : ∂L/∂h ← −λ · ∂L/∂h
```

**(3) Adversarial identity removal** (same mechanism, second nuisance factor):
```
L_id_adv = − Σ_{g}  1[identity(x)=g] · log A_i(GRL_λ(h))_g
```

**(4) Transliteration contrastive (InfoNCE)** — aligns Roman and Telugu script:
```
L_contrast = − log  exp( sim(z^R, z^N) / τ_c )
                  ────────────────────────────────
                   Σ_j exp( sim(z^R, z_j) / τ_c )
```

**(5) Script consistency** — the same comment must get the same answer in both scripts:
```
L_script_consist = KL( S(x^R) ‖ S(x^N) )
```

**(6) Fairness consistency** — swapping identity words must not change the answer:
```
L_fairness_consist = KL( S(x) ‖ S(x^I) )
```

### 3.5 Total objective
```
L = L_sent
  + λ₁ · L_script_adv
  + λ₂ · L_id_adv
  + λ₃ · L_contrast
  + λ₄ · L_script_consist
  + λ₅ · L_fairness_consist
```
Adversarial weights `λ₁, λ₂` are **ramped 0 → 1** over the first ~30% of steps for stability. Defaults: `γ=2`, `τ=1`, `τ_c=0.07`, `λ₃=λ₄=λ₅=1`.

### 3.6 Inference
```
p̄      = ½ ( S(x^R) + S(x^N) )          # script-TTA
T      = fit_temperature(p̄, dev)        # temperature scaling
p_cal  = softmax( log p̄ / T )
ŷ      = argmax p_cal    if  max p_cal ≥ θ
       = ABSTAIN         otherwise       # selective prediction
```

### 3.7 Why each term exists — and **which metric it moves**
| Mechanism (loss / step) | What it fixes | Metric it improves |
|---|---|---|
| `L_sent` (focal + logit adj.) | class imbalance | macro-F1, minority-class F1 |
| `L_script_adv` + `L_contrast` + `L_script_consist` | script reliance | **Δ_script**, macro-F1 |
| `L_id_adv` + `L_fairness_consist` | identity reliance | **identity-flip rate** (fairness) |
| script-TTA (`p̄`) | script noise | Δ_script, macro-F1 |
| temperature scaling | overconfidence | **ECE** |
| abstention threshold `θ` | unknown unknowns | **risk–coverage curve** |
| leakage-controlled split (C1) | evaluation validity | honest **F1 on Split B** |
| few-shot transfer (C6) | data scarcity | **transfer F1** |

> **Key point for reviewers:** every contribution is wired to a loss term and a measured number — no contribution is "just a claim."

---

## 4. End-to-end implementation (A → Z)

| Step | What you do | Tools | GPU? |
|---|---|---|---|
| 1 | Download **CMTET** (Telugu-English sentiment, ~19.9k) + optional backups | `datasets`, `git` | no |
| 2 | Minimal preprocessing (keep emojis/punct; keep raw copy) | `regex`, `pandas` | no |
| 3 | Confirm labels = core 3-class (pos/neu/neg) + aux | `pandas` | no |
| 4 | **De-duplicate** (exact + MinHash) → cluster IDs | `datasketch` | no |
| 5 | **Split**: (a) random 90/5/5; (b) grouped by cluster (**0% overlap**); (c) temporal if dates exist | `GroupShuffleSplit` | no |
| 6 | Generate `x^N` (Roman→Telugu script) for every item | IndicXlit | no |
| 7 | Build the **identity lexicon** (Telugu + Roman: caste/religion/region/gender) + generate `x^I` pairs | custom | no |
| 8 | Tokenize (max_len 96); build contrastive pairs | `transformers` | no |
| 9 | **Train SID-CM** (all 6 losses), 3 seeds | `peft`, `bitsandbytes`, `torch` | **yes** |
| 10 | **Evaluate**: macro-F1, per-class F1, MCC, accuracy, bootstrap CIs | `scikit-learn` | no |
| 11 | **Robustness**: Δ_script, script-TTA gain | IndicXlit | no |
| 12 | **Calibration**: temperature scaling → ECE; abstention curve | custom | no |
| 13 | **Fairness audit**: run all models on the probe set → flip rate | custom | no |
| 14 | Compare vs Telugu baselines; write up; release | — | no |

**Config (T4):** XLM-R-base / MuRIL, LoRA r=16 α=32; `max_len=96`; batch 16; lr 2e-4; 3 epochs; 3 seeds; λ ramp over 30% of steps.
**Local 4 GB fallback:** QLoRA 4-bit, batch 4 + grad-accum 8, gradient checkpointing.

---

## 5. Training time (GPU-hours)

| Job | Free T4 | RTX 2050 4 GB (QLoRA) |
|---|---|---|
| One SID-CM run (**Telugu ~19.9k**, 3 epochs) | **~1–3 h** | ~6–12 h |
| Main experiment (2 splits × 3 seeds) | ~8–18 h | too slow → T4 |
| Ablations (6 variants) | ~10–20 h | — |
| Calibration + robustness + fairness audit | < 1 h (CPU) | CPU |
| **Total** | **~20–45 GPU-h** | — |

→ Fits Kaggle's free **30 GPU-h/week**. Checkpoint every 250 steps + resume.

---

## 6. What we compare against (Telugu — no shared-task leaderboard exists)

| Target | Reported | Where |
|---|---|---|
| **CMTET baseline** (Kusampudi et al., RANLP 2021) | **~80.2% accuracy** (unsupervised normalization + MLP) | same split |
| **IEEE 2025 Telugu study** | 19,868 comments; transformer baselines | same split |
| Other Telugu code-mixed sentiment papers | varied (often accuracy on non-standard splits) | reproduced where possible |

> **Note:** Because there is **no Telugu sentiment shared task**, there is no standard macro-F1 leaderboard to beat. Compare against **individual Telugu papers** on the same split; the headline wins are on **Δ_script, ECE, abstention, identity-flip rate**, not raw F1.

**Win:** Δ_script · ECE · abstention · **identity-flip rate** · Split-B generalization · transfer F1.
**Tie:** raw macro-F1/accuracy on the random split. **Not claimed:** SOTA on an inflated split.

---

## 7. Step-by-step build plan (phases)

1. **P1 — Data + leakage study (CPU, first).** Build both splits; TF-IDF+SVM; compute the leakage gap.
2. **P2 — Views + counterfactuals.** Transliterate (`x^N`); build the identity lexicon; generate `x^I` pairs.
3. **P3 — Baseline model.** Plain XLM-R/MuRIL (LoRA), both splits, 3 seeds.
4. **P4 — Add SID-CM losses one at a time.** script-adv → contrastive → script-consist → identity-adv → fairness-consist.
5. **P5 — Inference upgrades.** script-TTA + temperature scaling + abstention.
6. **P6 — Robustness + cross-lingual.** Δ_script, few-shot transfer.
7. **P7 — Fairness audit.** Flip rate vs baselines.
8. **P8 — Ablations + significance + write-up + release.**

---

## 8. Why this is novel and publishable (honest)

- **Verified-open for Telugu:** no Telugu CMSA paper measures leakage, reports ECE, does script-adversarial de-biasing, does script-TTA, **or trains for identity-invariance** — and Telugu has no standard sentiment benchmark at all.
- **You gain a cleaner "first" claim** than you would for Tamil (where everything has been done).
- **The unifier is the strongest single idea:** one objective — *invariant to script and identity, sensitive to sentiment*.
- **Every contribution is measurable** (§3.7).
- **Feasible & reproducible** on free hardware (~20–45 GPU-h).
- **Venues:** DravidianLangTech → IEEE ICAAIC / IEEE Access.

**What we must NOT claim:** adversarial invariance (ADAN 2018), transliteration contrastive (TransliCo 2024), or identity-bias auditing (INDIC-BIAS) as inventions. Distinguish on *script + identity jointly, for Telugu, trained-in and calibrated*.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Leakage gap ~0 | Emphasize C2–C5; report the null result |
| No video IDs in CMTET → grouping weak | Use **cluster-based grouping** (still 0% near-dup overlap) |
| Adversarial heads hurt F1 | Ramp λ slowly; keep small; ablate |
| Contrastive needs big batches | Memory queue (MoCo-style) |
| Identity lexicon incomplete | Start with a small verified list; report coverage |
| Transliteration errors | Ensemble transliterators; report sensitivity |
| Only 1 local overnight run | Train on Kaggle/Colab; checkpoint + resume |

---

## 10. Key references (verified)
1. Kusampudi et al. 2021, *Sentiment Analysis in Code-Mixed Telugu-English Text with Unsupervised Data Normalization*, RANLP — **CMTET dataset (~19,857), ~80.2% accuracy baseline**.
2. IEEE 2025, *Sentiment Analysis of Telugu-English Code Mixed Social Media Text* (19,868 comments).
3. *Multilingual LMs Encode Script Over Linguistic Structure* (ACL 2026).
4. **TransliCo** (ACL 2024) — transliteration contrastive (prior art for C3).
5. **ADAN** (TACL 2018) — adversarial language invariance (prior art for C2/C5).
6. **SenTAS++** (IEEE 2025) — script-robust code-mixed sentiment, Marathi.
7. **Share What You Already Know** (ACM TALLIP 2024) — cross-script sentiment (Nepali/Hindi).
8. Dedup vs memorization (arXiv 2508.14062) — supports C1.
9. **INDIC-BIAS** (ACL 2025) / **IndiBias** (NAACL 2024) — identity-bias auditing precedent (C5).
10. Chakravarthi et al. 2022, *DravidianCodeMix*, LRE — context: **Telugu is absent** from it.

---

### The one-line pitch
> *Telugu code-mixed sentiment has no standard, leakage-controlled benchmark, and its models read the script instead of the meaning and may treat identity groups unequally. We train one model to be invariant to both — script (adversarial de-biasing + transliteration contrastive + consistency) and identity (adversarial de-biasing + counterfactual consistency) — then calibrate it, let it abstain, and evaluate it on the first leakage-controlled protocol for Telugu.*
