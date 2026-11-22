# ML-SGCA — Full Technical Pipeline & Research Specification
## Matrix Language Syntax-Guided Cross-Attention for Romanized Telugu–English Code-Mixed Sentiment

**Version:** 1.0  
**Date:** 2026-10-11  
**Primary dataset:** CMTE-NITANP (34,588 Romanized comments, 3-class) — evaluation on leak-free **Split B**  
**Held-out cross-corpus:** CMTET (19,868 comments)  
**Compute ceiling:** ≤ 2 GPU training runs (RTX 2050 4 GB, or free T4)  
**Companion:** `architecture/ML_SGCA_PITCH_DECK.md`

> **Verification policy:** every number labelled **[V]** is measured in this repo and traceable to a file under `Implementation/`. Every number labelled **[target]** is a design goal, not a result. Nothing here is reported as achieved until it is run.

---

## 1. Scope & Contribution Statement

We introduce **ML-SGCA**, the first sentiment model whose inductive bias is derived from **code-switching theory** (Myers-Scotton's Matrix Language Frame + 4-M model). It injects a **role-conditioned attention prior** into a frozen multilingual transformer so that **content morphemes cross-attend to the Matrix-Language system morphemes that scope them**.

**Three contributions, each tied to a metric:**
- **C1 — Structural diagnosis.** Show that the linear-beats-transformer anomaly aligns with an MLF prediction (sentiment is morpheme-local; the frame is noisy). *Metric:* polarity-shift coverage by system-morpheme features; error-subset analysis.
- **C2 — ML-SGCA architecture.** MLF/4-M role bias + content→system cross-attention over a frozen MuRIL backbone. *Metric:* Split-B Macro-F1 vs LogReg 0.8293 **[V]** and MuRIL 0.8167 **[V]**.
- **C3 — Automatic 4-M tagger.** Zero-manual-annotation morpheme-role tagging (LID + transliteration + Telugu FST + unsupervised clitic mining). *Metric:* tagger precision/recall on a human spot-check + sensitivity ΔF1.

---

## 2. Research Questions

- **RQ1 (Mechanism).** Do Matrix-Language system morphemes carry measurable, model-independent polarity signal in Romanized Tenglish?
- **RQ2 (Architecture).** Does an MLF/4-M role-conditioned cross-attention prior improve Macro-F1 over a strongly-tuned lexical baseline (0.8293 **[V]**) and a vanilla MuRIL (0.8167 **[V]**)?
- **RQ3 (Causality).** Is any gain attributable to the **MLF role structure** specifically, rather than to "more attention"? (Role vs random vs LID-generic bias ablation.)
- **RQ4 (Robustness/generality).** Does the MLF-typed model transfer zero-shot to an independent corpus (CMTET) and hold under orthographic perturbation?
- **RQ5 (Tagger robustness).** How sensitive is performance to automatic-tagger noise?

---

## 3. Data & Evaluation Protocol

| Item | Value |
|---|---|
| Primary | `datasets/splits/CMTE-NITANP/splitB_{train,val,test}.jsonl` — 27,670 / 3,459 / 3,459 |
| Class counts (test) | Negative 1,258 · Neutral 1,095 · Positive 1,106 |
| Leakage | 0.00% train-test twin overlap **[V]** (`datasets/splits/CMTE-NITANP/split_report.md`) |
| Control split | Split A (random) for the leakage sanity check |
| Cross-corpus | `datasets/processed/CMTET/labeled.csv` (19,868; 55.36% of multi-row clusters have conflicting labels **[V]**) |

**Rules (non-negotiable):** all hyperparameters/roles/thresholds tuned on `splitB_val` only; test touched once; fixed seeds; raw fields preserved (`text_raw`, `text_clean`) with added `roles`.

---

## 4. Mathematical Formulation

### 4.1 Notation
Comment `x = (w_1, …, w_L)`; encoder hidden `H ∈ R^{L×d}` (d=768); role of token `i` is `ρ(i) ∈ K = {C, ES, LSB, LSO, O}`.

### 4.2 Role embeddings
Add a learned role embedding to each token input: `e_i = e_tok(w_i) + e_role(ρ(i))`, `e_role ∈ R^{5×d_r}`.

### 4.3 MLF attention bias (typed, soft, learned)
For layer ℓ and head h:
```
M_ij^(h)   = b^(h)_{ρ(i), ρ(j)} + γ^(h) · A_ij
b^(h)      = b + u_h v_hᵀ                  (shared K×K MLF prior + per-head rank-1)
A_ij       = 1[ system morpheme j is hosted by / clitic-attached to content token i ]
α_ij^(h)   = softmax_j ( (q_i^(h)·k_j^(h))/√d_h + M_ij^(h) )      (padding −∞)
```
**MLF prior initialization of `b`** (rows = query role, cols = key role), a small hand-set matrix refined by training:
```
            C      ES     LSB    LSO    O
   C      0.0    +0.2   +1.0   +1.0   0.0     # content attends to its system morphemes
   ES    +0.2     0.0   +0.2   +0.2   0.0
   LSB   +1.0    +0.2    0.0   +0.2   0.0     # clitics anchor back to host
   LSO   +1.0    +0.2   +0.2    0.0   0.0     # negation/tense anchors to host
   O       0.0    0.0    0.0    0.0   0.0
```
Parameter count: `|b| = 25`, rank-1 per head `= 2·K·H_heads = 2·5·12 = 120`. **Total ≈ 145.** Negligible.

**Soft, not hard.** A hard structural mask is the limit `b → −∞`; we train the soft bias and report the hard mask only as ablation **A4** (§9). Rationale: automatic tags are noisy; a soft, MLF-initialized, differentiable bias is ablatable and self-correcting.

### 4.4 MLF Cross-Attention (C2 core)
Partition tokens into content `𝓒` and system `𝓢 = ES ∪ LSB ∪ LSO`. For content token `i`:
```
s_ij   = (Wq h_i)·(Wk h_j)/√d + β_{ρ(i),ρ(j)} + γ A_ij ,      j ∈ 𝓢
c_i    = Σ_{j∈𝓢} softmax_j(s_ij) · Wv h_j
h̃_i   = LN( h_i + λ · c_i )
```
Optional symmetric update for system tokens:
```
h̃_j   = LN( h_j + λ' · Σ_{i∈𝓒} softmax_i(s_ij) · Wv' h_i )
```
Intuition: this is the differentiable form of *"a Matrix-Language system morpheme scopes the polarity of its host content morpheme."*

### 4.5 Pooling & head
Attention-pool over content tokens with a learned query `q_0`:
```
a_i = softmax_i( (q_0 · h̃_i)/√d ),   z = Σ_{i∈𝓒} a_i h̃_i ,   p = softmax(W z + b)  ∈ R^3
```

---

## 5. Loss Functions

Primary:
```
L_sent = CE(p, y)
```
Optional, for imbalance (classes are near-balanced here, so off by default; ablate):
```
L_focal(focal CE + logit adjustment) = −Σ_k α_k (1−p_k)^γ y_k log p_k,  ℓ_k ← ℓ_k + τ log π_k
```
Optional scope auxiliary (supervises the attachment map; **weak/circular by construction** → ablate or drop):
```
L_scope = BCE( σ(s_ij), A_ij )         over (i∈𝓒, j∈𝓢)
```
Total:
```
L = L_sent + μ · L_scope        # μ = 0 by default; L_scope used only in the ablation grid
```
**We deliberately use no perturbation-consistency term** (UDA/VAT/JS are preempted and circular); the MLF bias is the sole non-standard inductive bias.

---

## 6. 4-M Role Tagger & Clitic Mining (automatic, no manual annotation)

```
Romanized token
   │ (1) LID
   ├─► IndicLID  |  or char n-gram LID trained on CMTET word-level tags, transferred
   │        → source ∈ {TE, EN, OTHER, MIXED}
   │ (2) transliterate Roman→native + segment
   ├─► IndicXlit / Aksharamukha  →  Telugu abugida
   ├─► Telugu morphological FST / segmenter  → morphemes + category
   │ (3) map categories → 4-M role ρ
   │        content (noun/verb/adj/adv, EN content)      → C
   │        plural/determiner inflection                → ES
   │        case/clitic (-lo,-ki,-tho,-ga)              → LSB
   │        tense/aspect/negation/agreement             → LSO
   │         (-ledu,-thondi,-kottindi, kaadhu, kani, asalu)
   │ (4) expand + absorb noise
   └─► seed lexicon (~50–100 forms) ⊕ unsupervised char-similarity clustering
       (-ledu/-ladu/-ledhu; kaadhu/kadu)  ⊕  lightweight learned role classifier
```
**Output per token:** `ρ(i)` and `A_ij` attachment links (same orthographic word / clitic-host).

**No manual bottleneck** — the seed list is tiny and public; expansion is unsupervised; final labels come from a **learned classifier** weak-supervised by the rules, so tag errors degrade gracefully and are absorbed by the soft bias.

**Tagger validation (review-critical, cheap):**
- Spot-check agreement on ~250 tokens (1 annotator; report % agreement).
- **Sensitivity:** shuffle/zero role tags → ΔF1 (ablation A7). If ΔF1 ≈ 0, the bias is not doing work (bad sign).

---

## 7. Adapter Implementation

**Backbone:** `google/muril-base-cased`, frozen.  
**LoRA:** `r=16, alpha=32, dropout=0.1, target_modules=["query","value"]` (extend to `["query","key","value","dense"]` in an ablation), `task_type=SEQ_CLS`.

```python
# pseudocode — single shared backbone, memory-neutral typed path
H = muril_lora(x.input_ids, x.attention_mask)          # B,L,d ; LoRA trainable, base frozen
H = H + role_embedding(rho)                            # e_role
for layer in cross_attn_layers:                        # 1 block
    H = mlf_self_attention(H, bias=M(rho, A))          # §4.3
    H = mlf_cross_attention(H, roles=rho, attach=A)    # §4.4  content <- system
z = attention_pool(H, content_mask)                    # §4.5
p = classifier(z)                                      # Linear(d, 3)
```

**New trainable parameters:** LoRA (~0.59M) + role embedding (5×768) + `b`/rank-1 (~145) + one cross-attention block (~2.4M for 768-d q/k/v/o). Total ≈ 3M trainable / 238M.

---

## 8. Training Protocol on Split B

| Setting | Value |
|---|---|
| Max length | 64 |
| Batch | 16 × grad-accum 2 (effective 32) |
| Precision | bf16 autocast (fp32 master weights) |
| Optimizer | AdamW, lr 4e-4, weight decay 0.01 |
| Schedule | linear warmup 6% → linear decay |
| Epochs | 8 |
| Seeds | 42 (primary); ≥5 seeds for final table |
| Selection | best val Macro-F1; test untouched until the end |
| Base | MuRIL-base full FT (no LoRA) → **GATE** |

**Measured VRAM envelope:** 1.58 GB / 3.87 GB at len 64, batch 16×2, bf16, LoRA **[V]**. ML-SGCA adds the role embedding + bias + one cross-attention block → memory-neutral. Headroom ≈ 2.2 GB → could raise batch to 24 or len to 96 (tight). Add gradient checkpointing only if len ≥ 96 or a second encoder pass is introduced.

**Runtime estimate:** ~45–55 min for Split A+B per full run (measured MuRIL+LoRA ≈ 45 min **[V]**).

**Stage plan:**
- **S0** tagger (CPU, ~30 min).
- **S1 GATE** full fine-tune MuRIL (1 GPU run). If full-FT already > 0.8293 → the anomaly is LoRA-specific; reframe to *efficiency + structural bias* (still publishable).
- **S2** ML-SGCA (1 GPU run).
- **S3/S4** ablations + CMTET + diagnostics (reuse checkpoints / CPU).

---

## 9. Diagnostic Ablations

| ID | Configuration | Purpose |
|---|---|---|
| A1 | Bias = 0 (plain LoRA) | lower bound; equals MuRIL 0.8167 **[V]** |
| A2 | Random role assignment (shuffle ρ) | is it the *role structure* or the *extra attention*? |
| A3 | Generic LID bias (binary TE/EN, not 4-M) | does 4-M beat a coarse LID prior? |
| A4 | Hard structural mask (b → −∞) | soft vs hard |
| A5 | Cross-attention removed (self bias only) | isolate C2 |
| A6 | Self bias removed (cross-attention only) | isolate §4.3 |
| A7 | Tagger shuffle/zero (sensitivity) | RQ5 |
| A8 | Full MuRIL fine-tune (no ML-SGCA) | gate control |
| A9 | Learned soft roles (tagger-free) | remove tagger dependency |
| A10 | `L_scope` on/off | auxiliary value |

Each ablation reports Split-B Macro-F1 ± std and the paired significance vs A1.

---

## 10. Cross-Corpus Evaluation on CMTET

- Zero-shot: train on NITANP Split B → evaluate on all 19,868 CMTET rows.
- **Report stratified:** by duplicate-cluster size and by label-conflict status (CMTET has 55.36% conflicting-label multi-row clusters **[V**]) so domain shift is separated from label noise.
- Treat CMTET as a **transfer probe**, not gold data; never tune on it.

---

## 11. Metrics & Statistics

- **Primary:** Macro-F1. Secondary: accuracy, weighted-F1, per-class P/R/F1, MCC.
- ≥5 seeds → mean ± std; **bootstrap 95% CIs**; **paired significance** (McNemar / bootstrap) vs LogReg (0.8293 **[V]**) and MuRIL (0.8167 **[V]**).
- Optional: ECE / risk–coverage (a cheap reliability add-on).
- Orthographic stress: Δ = F1_clean − F1_perturbed on held-out natural variants.

---

## 12. Reviewer Defense Checklist (Q1)

- [ ] **Novelty boundary explicit** — cite & differentiate: automatic MLID (EMNLP 2024, speech), SenTAS++ (LID graph attention), MultiSwitchNet (pragmatics), CharBERT/CMSAOne (fusion), UDA/VAT/SCR (consistency).
- [ ] **Falsifiable mechanism** — A2 (random roles) and A3 (LID-generic) prove the MLF *role* structure is causal.
- [ ] **Tagger validity + sensitivity** (§6, A7).
- [ ] **Statistical rigor** — ≥5 seeds, CIs, paired tests, leak-free split, no test tuning.
- [ ] **Generality** — ≥2 datasets (NITANP + CMTET / a second Dravidian language).
- [ ] **Decisive margin** — clears seed noise (≥ +0.01–0.015) and reconciles the conflict with Rayala & Seshadri (Springer 2025).
- [ ] **Honest limitations** — MLF is idealized; Romanized clitics are noisy; report failure modes.
- [ ] **Reproducibility** — release tagger, splits, code, checkpoints.

**Biggest single rejection risk:** a marginal margin. Mitigation: the mechanism-ablation suite turns "another attention bias" into a structural contribution; if the margin is modest, reposition as a structural-analysis paper (workshop tier) rather than over-claim Q1.

---

## 13. Reproducibility & Deliverables

- Code: `Implementation/baseline/muril/` (existing baseline) + new `Implementation/ml_sgca/` (tagger + model + ablations).
- Artifacts: M-sgca results `.md`/`.json` mirrored to the repo's per-dataset results layout.
- Publish: tagger, splits, seeds, environment pins, checkpoints (HF Hub).

## 14. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| Gate S1 shows full-FT already > 0.8293 | Reframe to efficiency + structural bias; still novel |
| ML-SGCA margin marginal (+0.003) | Publish as structural analysis / workshop; do not over-claim |
| Tagger noisy | Soft bias absorbs noise; A7 sensitivity; LSRO-free variant (A9) |
| 4 GB OOM at len ≥ 96 | Keep len 64; gradient checkpointing; halve batch |
| "Just attention" objection | A2/A3 ablations |
| One dataset | Add CMTET + a second Dravidian language |

## 15. References

- Myers-Scotton, C. *Social Motivations for Codeswitching* / MLF model; Myers-Scotton & Jake, **4-M model**.
- *Methods for Automatic Matrix Language Determination of Code-Switched Speech*, EMNLP 2024.
- *Matrix Language Effects on Colloquial Code-Switched Speech*, ACM 2025.
- SenTAS++ (IEEE 2025); MultiSwitchNet (KBS 2024); CharBERT (COLING 2020); CharacterBERT (EMNLP 2020).
- CMSAOne (FIRE 2020, arXiv 2101.09004); UDA (EMNLP 2021); R-Drop (NeurIPS 2021); VAT (2017); SCR (arXiv 2501.17598).
- Jiang et al., *Parameter-Free Classification with Compressors*, ACL Findings 2023.
- Trailblazer (DravidianLangTech 2026); Rahul & Adebayo (DravidianLangTech 2026); Rayala & Seshadri (Springer 2025).
- Kusampudi et al. (RANLP 2021); MuRIL, Khanuja et al. (2021).
