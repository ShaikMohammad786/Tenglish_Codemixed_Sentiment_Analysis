# Final Direction — Romanized Telugu–English Code-Mixed Sentiment Analysis
## Decision, novelty boundary, and an executable paper + experiment plan (≤2 GPU runs)

**Date:** 2026-10-11
**Dataset:** CMTE-NITANP (34,588 Romanized comments, 3-class) — eval standard = leak-free **Split B**; held-out cross-corpus = CMTET (19,868)
**Compute ceiling:** 1–2 GPU training runs (RTX 2050 4 GB local, or free/cheap cloud T4)

---

## 0. Decision (TL;DR)

**Recommended: Option C-plus — a *method-led* paper whose substrate is the diagnostic (B) and whose headline is a lightweight, orthographic-robust fusion that breaks the 0.83 ceiling and is validated under a *real* spelling-variation stress test.**

- **Reject A (OR-MuRIL / UDA consistency).** Novelty ≡ UDA/VAT/R-Drop; the premise that motivated it is dead; it is likely to *still* lose to LogReg, turning the paper's headline into a dressed-up negative result.
- **Reject *pure* B (audit/benchmark).** A 2026 DravidianLangTech paper already does robustness auditing of code-mixed sentiment (Hinglish), and the field is saturated with "benchmark" papers. B is necessary as the *substrate*, not as the headline. (User preference: headline novelty must live in the model, not the evaluation.)
- **Adopt C, reframed** so it cannot be dismissed as "just an ensemble": motivate the fusion from the diagnostic, add a data-driven orthographic-consistency objective, and validate on a naturally-attested held-out stress set.

**One-sentence thesis:** *Romanized Tenglish sentiment is bottlenecked by orthographic lexical coverage, not semantic capacity — a small lexical channel fused into a frozen transformer recovers the gap at <2% of the parameters and degrades far less under real spelling variation.*

---

## 1. Verified evidence base (all reproducible in this repo)

| Model | Split B Macro-F1 | Acc | Params | File |
|---|---|---|---|---|
| TF-IDF (word+char) + LogReg | **0.8293** | 0.8291 | 0 | `Implementation/baseline/logistic-regression/results/CMTE-NITANP/` |
| TF-IDF + LinearSVC | 0.8248 | 0.8245 | 0 | `.../svm/results/CMTE-NITANP/` |
| MuRIL-base + LoRA (r16, 8 ep, len64) | 0.8167 | 0.8167 | 592k trainable / 238M | `.../muril/results/CMTE-NITANP/muril_results.md` |
| CNN+BiLSTM (MuRIL WordPiece) | 0.7945 | 0.7942 | ~1.2M | `.../cnn-bilstm/results/CMTE-NITANP/cnn_bilstm_results_wordpiece.md` |
| CNN+BiLSTM (word) | 0.7863 | 0.7858 | ~1.2M | `.../cnn-bilstm/results/CMTE-NITANP/cnn_bilstm_results.md` |

Leakage gap |SplitA−SplitB| < 0.01 → **leakage hypothesis dead**.
Four hypotheses eliminated: leakage, script disparity (0 Telugu chars in NITANP), Indic-pretraining paradox (MuRIL saw Romanized Wikipedia), subword fragmentation (MuRIL WordPiece *helped* CNN+BiLSTM).
**Central surviving anomaly:** a 0-parameter TF-IDF model beats a 238M-parameter MuRIL by **1.26 F1** on a leakage-free split.

---

## 2. Why this is publishable (novelty boundary — stated honestly)

**Prior art that already exists (must be cited, cannot be claimed):**
- CharBERT (COLING 2020) / CharacterBERT (EMNLP 2020) — char+subword fusion *inside* a transformer.
- UDA (EMNLP 2021) / VAT (2017) / R-Drop (NeurIPS 2021) — perturbation consistency.
- Kusampudi et al. (RANLP 2021) — *rule-based* Telugu normalization on CMTET.
- Rahul & Adebayo (DravidianLangTech **2026**) — robustness audit of **Hinglish** sentiment (XLM-R 99.7% in-dist → 42.5% adversarial). Proves the venue wants robustness studies; has **not** done Telugu nor the linear-beats-neural angle.
- Rayala & Seshadri (Springer **2025**) — "transformers improve over baselines on **CMTE-NITANP**." Our leak-free numbers **contradict** this; must be reconciled (see §6 Stage 0).
- IEEE 11346359 — ensemble on Telugu CM social-media text, ~83.4% accuracy.
- MultiSwitchNet (KBS 2024), HLM (EMNLP 2025), SenTAS++ (IEEE 2025), TransliCo (ACL 2024).

**What is genuinely ours (the delta we can defend):**
1. **The anomaly, measured leak-free on Tenglish** — a 0-param lexical model ≥ fine-tuned MuRIL, with a *mechanistic decomposition* of why (orthographic lexical coverage vs semantic capacity). No one has reported this for Telugu; the two published Telugu transformer papers claim the opposite under different protocols.
2. **A minimal, interpretable fix** — a gated fusion whose inductive bias is chosen *because of* the diagnostic, not by trial and error.
3. **A naturally-attested, human-verified spelling-variation stress set** for Tenglish (held out from training) — the non-circular robustness test the field lacks (the 2026 Hinglish paper used a synthetic adversarial set).

**Positioning line:** *"the model is deliberately simple; the contribution is the diagnosis that dictates it."*

---

## 3. Mechanism hypothesis (what we test, not assume)

N-gram models win because they are **sub-morphemically orthography-invariant**: `bagundi`/`bavundhi`/`baagundhi` share character n-grams, so their signal survives spelling drift; subword models fragment these into near-disjoint tokens, and MuRIL's Romanized pretraining is **formal/rule-transliterated Wikipedia**, not casual Tenglish. We test this with three controlled ablations (§6): (a) lexicon coverage under spelling perturbation, (b) n-gram order sensitivity, (c) tokenization scheme (word vs MuRIL WordPiece vs char) at matched architectures.

---

## 4. Architecture — OR-Fuse (provisional name)

Two branches, jointly trained end-to-end in **one** GPU run:

```
 x = "cinema chala bagundi"
   ├── Lexical branch: char 3–5-gram hashing embed bag  → mean-pool → h_lex ∈ R^d  (trainable, ~0.5M)
   └── Semantic branch: MuRIL-base (frozen) + LoRA r=16,α=32 → [CLS] → h_sem ∈ R^768
                    g = σ( W_g [h_sem ; h_lex] )                    (gate)
                    h = g ⊙ h_sem + (1 − g) ⊙ h_lex
                    p(x) = softmax( MLP(h) )                        (3 classes)
```

**Objective (learning only from data; no hand rules):**
```
L = CE(p(x), y) + λ1 · D_JS( p(x) ‖ p(x̃) ) + λ2 · CE(p_lex(x), y)
where x̃ = a spelling variant drawn from variant clusters mined from TRAIN only
      (char-embedding cosine / edit-distance clustering; NOT hand-written g↔v rules)
```
- `D_JS` = symmetric Jensen–Shannon divergence (consistency under meaning-preserving orthographic shift).
- `λ1, λ2` linearly warmed 0 → 1 over the first epoch.

**Why this is not "just an ensemble":** the lexical branch and the gate are *trained jointly* with the semantic branch under a consistency objective; the gate learns *when* to fall back to orthographic-lexical evidence. We still report a plain probability-average ensemble (LogReg + MuRIL) as an ablation baseline to show the joint+consistency training adds value beyond averaging.

---

## 5. Contributions → metric map (every claim is measured)

| Contribution | Metric it moves |
|---|---|
| C1 diagnostic (anomaly + mechanism) | Split-B macro-F1 table; **fertility/coverage vs error** correlation |
| C2 OR-Fuse | Split-B macro-F1 (target > 0.8293), trainable-param count (<2% of MuRIL) |
| C3 orthographic robustness | Δ = F1_clean − F1_perturbed on **held-out natural** variants; prediction-consistency rate |
| C4 generalisation | CMTET zero-shot macro-F1 & transfer drop |
| efficiency | params, GPU-minutes vs full fine-tune |

---

## 6. Experiment plan (≤2 GPU runs) with decision gates

**Stage 0 — GATE (1 GPU run): reconcile the anomaly.**
Full fine-tune MuRIL-base (no LoRA, or all-linear LoRA) on Split B, 3 seeds.
- **Gate A (anomaly survives):** full-FT ≤ ~0.83 → headline stands ("transformers do not beat a 0-param model here"). Proceed to Stage 1 as planned.
- **Gate B (anomaly is a LoRA artifact):** full-FT > 0.8293 → reframe to *"efficient adaptation vs full fine-tuning"* and make OR-Fuse the efficient model that *matches full-FT at <2% params*. C-plus still holds; only the headline sentence changes.
*(This gate also tests our contradiction with Rayala & Seshadri 2025 — we report their protocol vs ours.)*

**Stage 1 — OR-Fuse (1 GPU run).** Train the architecture in §4 on Split B, 3 seeds. Report vs all Split-B baselines + the plain-ensemble ablation.

**Stage 2 — Robustness & generalisation (CPU/GPU inference only).**
- Build the **natural spelling-variation stress set**: mine variant clusters from the corpus, sample held-out variants, human-verify label preservation (native/fluent annotator). Report Δ per model.
- Report clean vs perturbed F1 + prediction-consistency + bootstrap 95% CIs.
- Zero-shot CMTET (report cluster- and label-conflict-stratified metrics; note CMTET's 55% conflicting-label clusters).

**Stage 3 — Diagnostic ablations (CPU).** n-gram order sweep (1–6); word vs WordPiece vs char at matched capacity; fertility/coverage vs error.

**Total GPU: 2 runs** (full-FT + OR-Fuse). Everything else CPU/inference.

---

## 7. Target tables

1. **T1** Split-B in-domain: LogReg, LinearSVC, CNN+BiLSTM (word/WordPiece), MuRIL+LoRA, MuRIL full-FT, plain ensemble, **OR-Fuse** (macro-F1 ± std, acc, params).
2. **T2** Diagnostic: n-gram order / tokenization scheme / coverage-vs-error.
3. **T3** Orthographic stress (natural held-out): clean vs perturbed F1, Δ, consistency.
4. **T4** Zero-shot CMTET (cluster/label-conflict stratified).

---

## 8. Paper outline & venue

**Title (draft):** *"When Zero Parameters Beat 238 Million: Orthographic Robustness and a Lexical–Semantic Fusion for Romanized Telugu Sentiment."*

1. Intro — Tenglish; the leak-free anomaly; why it matters for low-resource Indic.
2. Related work — Dravidian CMSA; char/subword models; consistency training; robustness audits (name the 2026 Hinglish paper).
3. Benchmark integrity — Split B, 0.00% leakage; the leakage non-finding as hygiene; reconcile Springer-2025 NITANP claim.
4. The anomaly + mechanism (RQs).
5. OR-Fuse — formulation (§4).
6. Experiments — T1–T4; ablations; significance.
7. Robustness discussion + guidance.
8. Conclusion.

**Venue (honest):** primary target **DravidianLangTech** workshop (fast, high-fit, cheap/near-free) → extended to **IEEE Access** (gold OA, APC applies — verify current fee) or **ACM TALLIP** / **Springer SNAM** / **Natural Language Processing (Cambridge)**.

---

## 9. Risks & mitigations / fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Full-FT beats LogReg (anomaly is LoRA-specific) | Gate B reframe (efficiency story) — plan still works |
| OR-Fuse does not beat 0.8293 | Fall back to headline = diagnostic + robustness (B), method as demonstration; still publishable at the workshop |
| "It's just an ensemble" | Report joint-vs-probability-ensemble ablation; the joint+consistency must show a gain, else drop the claim honestly |
| Natural-variant set too small | Supplement with keyboard-edit-distance noise; report both |
| CMTET label noise | Cluster/label-conflict-stratified reporting; treat as probe not gold |
| Compute (4 GB) | bf16, batch 16×2, len 64, LoRA; already validated (MuRIL run fit in 1.58 GB) |

---

## 10. Verification / definition of done
- Split-B table reproduced for every baseline from committed scripts under one seed regime.
- Gate A/B decision recorded explicitly.
- OR-Fuse reported with mean ± std over ≥3 seeds, bootstrap 95% CIs, and a paired significance test vs LogReg and vs MuRIL.
- Stress set released with label-preservation verification and agreement reported.
- All numbers traceable to a file under `Implementation/`; no tuned-on-test.
