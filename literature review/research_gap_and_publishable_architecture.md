# The Critical Research Gap and a Publishable Novel Architecture
## For Code-Mixed Sentiment Analysis in Dravidian Languages (2026)

**Companion to:** `literature_review_dravidian_codemixed_sentiment_2020_2026.md`
**Date:** 28 September 2026
**Purpose:** Identify the single most consequential *unsolved* problem in Dravidian code-mixed sentiment analysis (CMSA), then specify a novel, feasible, publishable architecture + benchmark + evaluation protocol that is positioned to outperform every currently published system on the dimensions that matter.

> **UPDATED / SUPERSEDED IN PART.** A deeper, more detailed design — architecture **SAMVAD**, the 2025–2026 benchmark **DravidianBench-X**, dataset strategy (why the field's 2019 data is a liability), tokenizer design, MoE adapters, retrieval augmentation, and full methodology — is in **`SAMVAD_research_paper_full_methodology_architecture.md`**. Read that document first; the SCRIPT-D sketch below is retained as the conceptual precursor.

> **Honesty note first.** No research direction "guarantees" publication. What can be engineered is a *high probability of acceptance at a strong venue*: a gap that reviewers already agree is important, a method that is novel relative to prior art, an evaluation that is rigorous, and a contribution that survives even if the headline accuracy gain is modest. This document is engineered for exactly that, and §12 states the residual risks plainly.

---

## 0. Executive Summary

**The chosen gap (one sentence):** *Dravidian CMSA has no script-invariant, leakage-free, cross-family-transferable model — current published numbers are produced under a random, single-script, movie-trailer-only, evaluation regime that (a) risks near-duplicate train/test leakage and (b) collapses the moment the script, domain, or neighbouring Dravidian language changes.*

**The contribution (three artefacts):**
1. **DravidianBench-X (DBX)** — the first versioned, de-duplicated, **dual-script**, multi-domain, multi-language (Tamil, Malayalam, Kannada, Telugu, Tulu) CMSA benchmark with **leakage-controlled video/temporal splits** and a robustness suite (**Drav-SHIFT**).
2. **SCRIPT-D** — a **S**cript-**C**onsistent **R**epresentation with **I**ntra-Dravidian **P**rojection and **T**ransliteration-**D**irected contrastive learning: a language-tag- and script-aware multilingual encoder + light LID-guided graph attention + mixture-of-family-adapters + calibrated, imbalance-aware heads.
3. **A five-axis robustness protocol** (cross-script, cross-domain, cross-language, temporal, low-resource) with calibration (ECE) and significance testing baked in.

**Why it can beat the field:** it is the first system evaluated on *all four* shift axes at once with leakage removed; it introduces the exact inductive bias (script-invariance + family transfer) that current models lack; and — critically — it wins even where raw F1 ties, because it reports calibrated, statistically tested, per-shift results that no existing paper provides.

---

## 1. The Gap, Stated Precisely

### 1.1 Formal statement
Let a Dravidian code-mixed comment be realised in at least two scripts — Roman transliteration (`x_rom`) and native abugida (`x_nat`) — from one of several domains `d ∈ D`, one of several languages `ℓ ∈ L = {Tamil, Malayalam, Kannada, Telugu, Tulu}`, and one of several time periods `t`. Current models `f(x_rom; ℓ, d, t)` are trained and evaluated with **no invariance pressure across script**, on **one domain** (`d = movie-trailers`), for **one language at a time**, under **random comment-level splits** over **2019 data**.

Two things are therefore unsolved *and jointly necessary*:

- **(A) Invariance:** `f(x_rom) ≈ f(x_nat)` for the same underlying sentiment, and `f` degrades gracefully under domain/language shift.
- **(B) Measurement validity:** the reported numbers must not be inflated by near-duplicate leakage or rendered incomparable by metric/scheme drift.

You cannot claim (A) without fixing (B): if the test set contains near-copies of training comments, you cannot tell whether a model generalises or memorises, and a "script-robust" model may simply be reading leaked vocabulary.

### 1.2 Why this is *the* crucial gap
| Evidence | Source |
|---|---|
| Every core corpus is a 2019 YouTube **movie-trailer** crawl; cross-domain validity untested. | Dataset papers [V] |
| DravidianCodeMix uses a **random 90/5/5** split; YouTube comments on one trailer are near-duplicate spam. Near-duplicate reuse is a documented benchmark-leakage mechanism. | Split table in FIRE 2021; leakage literature [V] |
| Corpora contain Roman **and** native script, yet models are almost always trained/tested in **one** script. | Dataset descriptions [V] |
| The field's only substantive architectural paper for Dravidian (MultiSwitchNet) models pragmatics, **not** script invariance; the script-invariance techniques that do exist (TransliCo, SenTAS++) are **not** applied to any Dravidian language. | [V] |
| Kannada froze in 2021, Telugu never had a comparable SA leaderboard, Tulu is tiny; **no cross-Dravidian transfer study** exists. | §6 of the review [V] |
| The 2026 workshop **dropped the core SA task entirely** — a symptom that the generic benchmark line has saturated. | DravidianLangTech-2026 [V] |

### 1.3 What the field itself says
The PDF seed document's own conclusion: *"the most important unresolved problem is generalisation across scripts, transliteration styles, domains, dialects, time periods, and language pairs."* The survey literature concurs: LLMs "still struggle with mixed-language inputs, limited CSW datasets, and **evaluation biases**" (ACL 2026 survey). The gap is not my invention — it is the consensus open problem. What is missing is a *focused, feasible architecture + benchmark that closes it.*

---

## 2. Why Other Candidate Gaps Are Weaker (Ranking)

| Rank | Candidate gap | Importance | Novelty | Feasibility | Verdict |
|---|---|---|---|---|---|
| **1** | **Script-invariant + leakage-free + cross-Dravidian generalisation** | Very high | High | **High** (data exists; techniques proven on other languages) | **CHOSEN** |
| 2 | Sarcasm / implicit sentiment | High | Medium-high | Medium (labelled data small) | Strong *second paper*; make it an auxiliary head in #1 |
| 3 | Multimodal (text+audio) Dravidian SA | High | High | Low (data scarce; 2022 task had 1 team) | High-risk; not a first contribution |
| 4 | Heterogeneous graph transformer + federated + multitask (seed Doc C's plan) | High | High | **Low** (5 languages × 6 heads × federated = over-scoped; likely unfinished) | Too broad; subsumed as an *optional* module |
| 5 | LLM fine-tuning for Dravidian CMSA | Medium | Low | High | Crowded; incremental ("yet another LLM fine-tune") |
| 6 | Unified benchmark *alone* | High | Medium | High | Publishable but "just a dataset"; pair with a method |
| 7 | ABSA / aspect-level | High | High | Low | Data almost absent for code-mixed Dravidian |
| 8 | Calibration / explainability | Medium | High | High | Best *reported* as part of #1, not standalone yet |
| 9 | Federated privacy-preserving | Medium | High | Medium | Interesting extension after #1 |
| 10 | Emotion / hope-speech / AI-review detection | Medium | Medium | Medium | Adjacent tasks; not the core SA problem |

**Design principle:** pick the gap where *the data already exists, the techniques are proven elsewhere, and no one has combined them for Dravidian* — that maximises the ratio of novelty to execution risk. That is #1.

---

## 3. The Research Programme — Three Contributions

### C1 — DravidianBench-X (DBX): a leakage-free, dual-script, multi-domain, 5-language benchmark
- Harmonises Tamil/Malayalam/Kannada/Telugu/Tulu sentiment into a **common core scheme** (Negative / Neutral / Positive) with **preserved auxiliary labels** (mixed, offensive, sarcasm) so nothing is lost for multi-task use.
- Provides **official grouped splits** (by video ID and time), a **de-duplicated "clean" test set**, and the **Drav-SHIFT** suite (§6.3).
- Ships **parallel Roman↔native renderings** for a verified subset and a controlled **transliteration-perturbation** test set.

### C2 — SCRIPT-D: the model (§4)
A script-consistent, family-transferable, imbalance-calibrated architecture.

### C3 — The five-axis robustness protocol (§7)
Cross-script · cross-domain · cross-language · temporal · low-resource, each with macro-F1, per-class F1, MCC, **ECE**, bootstrap CIs, and paired significance tests.

Each is individually publishable; together they form a thesis-grade, high-impact programme.

---

## 4. Proposed Architecture — SCRIPT-D

### 4.1 Design principles (traceable to the gap)
1. **Script invariance must be learned, not preprocessed away** (translation destroys switch structure; §5.3 of the review). → contrastive, not translational, alignment.
2. **Language identity and switch points are signal, not noise.** → LID-guided attention.
3. **Dravidian languages share typology.** → shared family adapter + per-language adapters.
4. **The label distribution is 56/40/46% majority-positive.** → imbalance-aware, calibrated heads.
5. **Everything must be ablatable** so the paper can isolate which component delivers the gain.

### 4.2 Pipeline

```
                       ┌───────────────────────── Input views ─────────────────────────┐
  raw comment x ──►  ├─ x_rom        (original Romanised CM text)
                     ├─ x_nat        (transliterated to native script; IndicXlit + rules)
                     └─ x_pert       (spelling/script perturbations for robustness)
                       │
       ┌───────────────┴──────────────── shared multilingual encoder E (XLM-R-large / MuRIL) ───────────┐
       │  token embeddings  +  LID-type embedding  +  script-type embedding  +  switch-boundary flag     │
       └───────────────┬─────────────────────────────────────────────────────────────────────────────────┘
                       │  contextual token states H
                       ▼
        ┌───────────────────────────────┐        ┌──────────────────────────────────────┐
        │  LID-GAT (optional, 1–2 layers)│        │  Mixture-of-Family-Adapters (MoFA)   │
        │  typed edges: adjacency,       │        │  shared Dravidian adapter +          │
        │  dependency, switch-boundary,  │        │  per-language adapters, routed by ℓ  │
        │  transliteration-equivalence,  │        └──────────────────────────────────────┘
        │  negation                      │
        └───────────────┬───────────────┘
                        ▼  fused representation z
   ┌────────────────────┼───────────────────────────────┬──────────────────────┐
   ▼                    ▼                                ▼                      ▼
 Sentiment head    Auxiliary heads                 Contrastive projector   Calibration
 (core 3/5-class,  LID, script, offensive,          (InfoNCE on z)          temperature/
  class-balanced   sarcasm                                    │            Dirichlet
  focal + logit)                                                          (post-hoc)
```

### 4.3 Component detail

**(1) Input views & alignment.** Generate `x_nat` from `x_rom` with an ensemble (IndicXlit + rule-based Roman→abugida mapping) and, where possible, human-verified pairs. Keep token offsets so LID and switch features transfer across views. Generate `x_pert` by (i) random transliteration of a fraction of tokens, (ii) script conversion, (iii) spelling noise — used only for robustness evaluation, never for training leakage.

**(2) Shared encoder + typed embeddings.** Use a multilingual encoder `E` (start with **XLM-R-large** and **MuRIL**; both are strong in the verified leaderboards). Concatenate/add four learned embeddings per token:
- `e_tok` (subword),
- `e_lang ∈ {ta, ml, kn, te, tu, en, other}` from the LID module,
- `e_script ∈ {native, roman}`,
- `e_switch ∈ {0,1}` (is this token at/adjacent to a switch boundary?).

**(3) LID-guided graph attention (LID-GAT).** Build a token graph `G=(V,E,R)` with **typed** edges `R = {adjacency, dependency, switch-boundary, transliteration-equivalence, negation}`. Apply relational attention with per-edge-type bias (à la RGT-style decoupling). This is deliberately **lightweight (1–2 layers)** so it is (a) trainable on small data and (b) ablatable — avoiding the over-scoped heterogeneous-graph programme of seed Doc C.

**(4) Transliteration-directed contrastive learning (the core novelty).** Project pooled `z` to a unit sphere. InfoNCE with:
- **Positives:** the *same comment* in `x_rom` and `x_nat`; the same comment under benign perturbation.
- **Hard negatives:** (i) negation-flipped pairs ("*did* like" vs "*didn't* like"), (ii) sarcastic-vs-literal pairs sharing vocabulary (e.g. positive English slang in a negative Tamil frame), (iii) different comments sharing sentiment words.
This is the first application of **TransliCo-style script-barrier contrastive learning to code-mixed sentiment**, and the hard-negative design (negation + sarcasm) is the novel twist that makes it a *sentiment* objective rather than a generic alignment objective.

**(5) Mixture-of-Family-Adapters (MoFA).** A **shared "Dravidian" adapter** captures family-level features; **per-language adapters** capture language specifics; a small router `g(ℓ)` mixes them. Because Tamil/Malayalam/Kannada have data and Tulu/Telugu less, MoFA is the mechanism for **Tamil→Tulu** and **Telugu→Kannada** transfer. (This adapts the AdapterFusion-for-code-mixed idea to a *Dravidian family + script* setting — new.)

**(6) Heads & losses.**
```
L = λ₁·L_sent  +  λ₂·L_LID  +  λ₃·L_script  +  λ₄·L_off  +  λ₅·L_sarc  +  λ₆·L_contrast  +  λ₇·L_consistency
```
- `L_sent`: **class-balanced focal loss** with **logit adjustment** (prior-corrected) to fight the 8:1/2:1 imbalances without discarding data.
- `L_LID`, `L_script`, `L_off`, `L_sarc`: auxiliary supervision from existing annotations (DravidianCodeMix word-level LID, offensive labels, sarcasm set).
- `L_consistency`: KL between `p(y|x_rom)` and `p(y|x_nat)` (a cheap, direct invariance regulariser).
- Post-hoc **temperature scaling / Dirichlet calibration** on a held-out split; report **ECE**.

**(7) Training curriculum (keeps it tractable).**
- Stage 0 — domain-adaptive MLM on unlabelled Dravidian CM text.
- Stage 1 — contrastive + LID/script pretraining on auto-generated Roman↔native pairs.
- Stage 2 — multitask supervised fine-tuning with **LoRA on adapters** (cheap, reproducible).
- Stage 3 — cross-lingual transfer: train on {Tamil, Malayalam, Kannada}, target {Tulu, Telugu} zero/few-shot; leave-one-language-out.
- Stage 4 — calibration + per-class threshold tuning on dev.

### 4.4 Compute
Single-GPU-feasible: LoRA/adapters on XLM-R-large (~560M) or MuRIL (~236M) plus a 1–2 layer LID-GAT. No model exceeds ~1B trainable-adjacent parameters, so the study is reproducible on a 24–48 GB GPU — an explicit advantage over "fine-tune a 70B LLM" approaches and a point reviewers reward.

---

## 5. Novelty Analysis vs Prior Art (this is what a reviewer will check)

| Prior work | What it did | What SCRIPT-D adds |
|---|---|---|
| **TransliCo** (ACL 2024) | Contrastive learning to break the **script barrier** in multilingual PLMs (general). | Applies script-barrier contrastive learning **to code-mixed sentiment**, with **sentiment-specific hard negatives** (negation, sarcasm). |
| **SenTAS++** (IEEE 2025) | Transliteration + token-level LID + cross-view contrastive fusion + language-tag **graph attention** — for **Marathi**. | Ports the *idea* to the **Dravidian family**, adds **family adapters + cross-lingual transfer + calibration**, and a **leakage-free multi-script benchmark**. First for Dravidian. |
| **MultiSwitchNet** (KBS 2024) | Socio-pragmatic framework for Dravidian-English CM SA. | Different axis (pragmatics→script invariance); SCRIPT-D's contrastive + family-transfer is orthogonal and can be combined. |
| **CDMG-BERT** (Hinglish) | Domain-adaptive multitask graph-enhanced BERT for **Hinglish**. | First graph-enhanced + contrastive system for **Dravidian**; explicit **cross-Dravidian** transfer. |
| **AdapterFusion for code-mixed** (EAAI 2023) | Language-based AdapterFusion, general. | **Family-shared + language-specific adapters** with a router, targeted at **intra-Dravidian low-resource transfer**. |
| **GCN+multi-head attention for CM SA** (2021) | Only Dravidian graph attempt; proof of concept. | Typed, LID-guided graph attention integrated with contrastive invariance and calibration. |
| **HLM** (EMNLP 2025) | Encoder+small-decoder to match LLMs on CM sentiment. | Complementary (efficiency) vs SCRIPT-D (robustness/generalisation). |
| **All shared-task systems** | Single script, single domain, random split, one metric. | Dual script, multi domain, grouped/leakage-free split, five axes, calibrated, significance-tested. |

**Novelty statement (one line):** *SCRIPT-D is the first cross-Dravidian, script-invariant, imbalance-calibrated sentiment architecture, and DBX is the first leakage-controlled, dual-script, multi-domain Dravidian benchmark — neither the model combination nor the evaluation regime exists in the published Dravidian literature.*

---

## 6. Dataset & Benchmark Plan (DBX)

### 6.1 Sources and reuse (all public)
| Component | Source | Use |
|---|---|---|
| Tamil/Malayalam/Kannada core | DravidianCodeMix (44,020 / 19,616 / 7,671) | Core SA + LID + offensive aux |
| Tamil/Tulu (newer, larger) | DravidianLangTech 2023–2025 corpora (Tamil 33,989+3,786 train/dev; Tulu 6,458+781) | Core SA; **Tulu transfer target** |
| Tamil emotion | TamilEmo (>42k, 3/7/31 classes) | Optional emotion aux head |
| Sarcasm | Tamil+Malayalam sarcasm set (SNAM 2025) | Sarcasm aux head + hard negatives |
| Political (2026) | TamilPoliSent (5,440 tweets, 7 classes, 8:1) | Cross-domain + fine-grained subset |
| ABSA | MADTRAS (Tamil, 4,000 reviews/7,507 aspects) | Optional ABSA probe |
| **New crawl** | News, product reviews, forums, other Dravidian content | **Enables cross-domain + Telugu SA** |

### 6.2 New annotation (the only major cost)
- **Telugu sentiment** (to give Telugu a comparable SA resource) — modest target (~8–10k comments), 3 annotators, native-speaker guidelines.
- **Cross-domain sample** across Tamil/Malayalam/Kannada (~5k each).
- **Dual-script verification** of a ~3–5k subset (Roman↔native pairs).
- Report **Krippendorff's α** (target ≥0.7, matching the field's norm).
- **Ethics:** public YouTube/X data, remove PII and usernames, publish annotation guidelines + datasheet, respect source licences, document annotator consent/pay.

### 6.3 Drav-SHIFT robustness suite
Auto-generated, no new annotation:
1. **Script shift:** Roman→native, native→Roman, mixed→single via transliteration.
2. **Spelling noise:** character substitutions, elongation (`suuuuper`), abbreviations.
3. **Emoji/punctuation removal.**
4. **Switch-point corruption:** random re-ordering of language blocks.
5. **Adversarial switching:** swap a sentiment-bearing token with an opposite-polarity synonym across languages.

---

## 7. Experimental Design

### 7.1 Baselines (must all be reproduced on DBX)
Majority-class predictor; TF-IDF + LR/SVM; TF-IDF + ensemble (reproduce the Kannada-winning regime); fastText; BiLSTM; CNN+BiLSTM; mBERT; XLM-R-base/large; IndicBERT; MuRIL; Tamil-sentiment-BERT; transformer ensemble (soft voting); +word-level LID feature (Chanda et al.); MultiSwitchNet-style pragmatic baseline; a LoRA-tuned XLM-R; a **zero/few-shot open-weight LLM** (Llama-3.1-8B / Gemma-2-9B) for the LLM-comparison row; and **SCRIPT-D / SCRIPT-D w/o graph / w/o contrastive / w/o MoFA**.

### 7.2 Five evaluation axes
- **In-domain:** official DBX test (reproduces shared-task setting).
- **Cross-script:** train Roman → test native and vice versa.
- **Cross-domain:** train movie-trailers → test politics/news/product.
- **Cross-language:** leave-one-language-out; zero-shot and 5/10/25% few-shot transfer (Tamil→Tulu, Telugu→Kannada).
- **Temporal:** train ≤2022 → test ≥2024 (slang drift).
- **Low-resource:** 1 / 5 / 10 / 25 / 100% training.

### 7.3 Metrics
Macro-F1 (primary), weighted-F1, accuracy, **per-class P/R/F1**, **MCC**, **ECE + reliability diagrams**, **cross-script robustness drop** `Δ = F1_in − F1_shifted`, and **leakage gap** `F1_random-split − F1_grouped-split`.

### 7.4 Statistics
≥5 seeds; **bootstrap 95% CIs**; **paired significance tests** (bootstrap/McNemar) between SCRIPT-D and each baseline; report effect sizes. This alone differentiates the work from ~all existing Dravidian CM-SA papers, which report point estimates only.

### 7.5 Ablations
Remove, one at a time: LID embedding, script embedding, switch feature, LID-GAT, transliteration view, contrastive loss, negation/sarcasm negatives, MoFA (family vs per-language vs single), focus loss, logit adjustment, calibration. Report each on **all five axes** (ablation × robustness grid).

---

## 8. Why This Can Beat All Current Papers (mechanistic, falsifiable)

"Beats" is defined as: **higher macro-F1 on a leakage-controlled benchmark AND smaller degradation under script/domain/language/time shift AND better-calibrated outputs — reported with statistical significance.** Five falsifiable predictions:

1. **Leakage gap is real (predicts baselines *fall*).** On grouped/de-duplicated splits, existing SOTA scores will drop measurably relative to their published random-split numbers. → *If true, every prior Dravidian CM-SA leaderboard number becomes qualified — a headline result.*
2. **Script invariance.** SCRIPT-D will reduce the cross-script macro-F1 drop by a **large relative margin** versus best transformer baselines, because only SCRIPT-D has an explicit invariance objective and a consistency loss. → *Falsifiable by the Δ metric; the mechanism (contrastive + KL) is causal, not incidental.*
3. **Cross-Dravidian transfer.** MoFA will beat single-language fine-tuning on **Tulu and Telugu** few-shot, because family-level features are shared and target data is scarce.
4. **Minority classes & calibration.** Class-balanced focal + logit adjustment + post-hoc calibration will improve **minority-class recall and ECE** at equal macro-F1 — which matters more than aggregate F1 in real moderation.
5. **Completeness.** DBX is the first system-level comparison across **5 languages × 2 scripts × 4 shift types**, so SCRIPT-D is the only entry that can be *ranked on every axis* — structurally dominating papers that report one number on one dataset.

**Even in the pessimistic case** (SCRIPT-D ties on in-domain F1), contributions C1 (benchmark + leakage finding) and C3 (rigorous protocol) remain independently publishable, and the null/partial results are themselves informative — which is precisely why this direction has a high acceptance probability rather than a lottery's.

---

## 9. Feasibility, Timeline, Risks

### 9.1 Feasibility
- **Data:** 95% public (only the Telugu + cross-domain crawl is new annotation).
- **Compute:** ≤1B-param models + LoRA → one modern GPU; no LLM training.
- **Skills:** standard transformer fine-tuning + contrastive learning + graph attention; all with mature libraries (HuggingFace, PEFT, PyTorch Geometric).
- **Tooling:** IndicXlit/IndicLID for transliteration & LID (public).

### 9.2 Indicative milestones (no calendar promises — sequence only)
| Phase | Deliverable |
|---|---|
| P1 | Corpus audit; reproduce baselines on original splits; **compute the leakage gap** |
| P2 | Release **DBX v1** (harmonised labels, grouped splits, dedup, dual-script subset) |
| P3 | Reproduce top-5 historical systems on DBX (fair, metric-consistent leaderboard) |
| P4 | **SCRIPT-D v1** (encoder + LID/script features + contrastive); first cross-script results |
| P5 | Add LID-GAT + MoFA; cross-language transfer experiments |
| P6 | Calibration + full five-axis + ablations + significance testing |
| P7 | Write-ups (§10), data/code release, leaderboard |

### 9.3 Risks & mitigations
| Risk | Mitigation |
|---|---|
| Cross-lingual transfer gains are small | Report as a *finding*; frame benchmark + leakage as the primary contribution |
| Transliteration tools are noisy on Tulu | Ensemble + rule-based fallback; treat transliteration noise as part of the robustness study |
| New annotation cost | Start with Telugu only; expand conditionally; reuse existing high-α labels |
| Test-set contamination by LLM-augmented data in future | Publish a frozen, timestamped DBX test set (a "living benchmark" is itself a contribution) |
| Ethical/legal (YouTube/X) | Public data, PII scrubbing, datasheet, licence compliance, no redistribution of removed content |
| Reviewer: "this is just TransliCo for Dravidian" | The contribution is the *combination + benchmark + five-axis protocol + leakage finding*; pre-empt in related-work table (§5). |

---

## 10. Publication Plan (decompose into three papers)

1. **Paper A — Benchmark + validity study (resource/analysis).**
   *"DravidianBench-X: A Leakage-Controlled, Dual-Script Benchmark Reveals the Generalisation Illusion in Dravidian Code-Mixed Sentiment Analysis."*
   Targets: **LREC-COLING**, ACL/EMNLP (resource or analysis track), *Language Resources & Evaluation*, *Social Network Analysis & Mining*.
2. **Paper B — Method.**
   *"SCRIPT-D: Transliteration-Contrastive, Family-Adaptive Sentiment Analysis for Dravidian Code-Mixed Text."*
   Targets: **ACL / EMNLP / NAACL** (main or Findings), *Natural Language Processing* (Cambridge), *Knowledge-Based Systems*, *IEEE Access*.
3. **Paper C — Transfer + calibration (or a workshop short paper first).**
   *"Cross-Dravidian Transfer and Calibration for Low-Resource Code-Mixed Sentiment."*
   Targets: **DravidianLangTech** (workshop — fast, high-fit), *IEEE Access*, *Scientific Reports*.

**Ordering advice:** publish the workshop short paper (C-mini) first for a citable record, then A, then B. Reviewers of B will already cite A.

---

## 11. Immediate Action Checklist (next steps)
- [ ] Download DravidianCodeMix + Tamil–Tulu 2023–2025 corpora; verify counts; build the de-duplication pipeline (MinHash/SimHash).
- [ ] Re-split with **video-ID grouping**; measure the leakage gap on an mBERT/XLM-R baseline (this alone is Paper A's core experiment).
- [ ] Stand up the representation pipeline: LID (IndicLID) + transliteration (IndicXlit) + script view generation.
- [ ] Implement SCRIPT-D stage 1 (contrastive + LID/script) and reproduce an XLM-R baseline on the same splits.
- [ ] Design the Telugu annotation guideline + ethics/datasheet template.
- [ ] Set up the metrics harness: macro-F1, per-class, MCC, ECE, bootstrap CIs, McNemar.
- [ ] Pre-register the five-axis evaluation and ablation grid (reduces p-hacking risk; reviewers like it).

---

## 12. Threats to Validity & Honest Caveats
1. **"Guaranteed publishable" is not achievable**; the claim here is *high probability* due to a consensus gap + proven techniques + rigorous protocol + multiple independent contributions.
2. **Leakage may be smaller than expected.** If the leakage gap is negligible, Paper A's headline weakens — but the benchmark and protocol remain valuable. The experiment is cheap, so *test it early* (P1).
3. **Script invariance could be partly solved by transliteration alone.** Mitigate by ablating transliteration-only vs contrastive; the contrastive term must show incremental value or it is dropped honestly.
4. **Telugu/Tulu data scarcity** may cap transfer gains; position transfer as exploratory.
5. **LLM baselines are a moving target;** use open-weight models and pin versions for reproducibility.
6. **Annotation subjectivity** (mixed/neutral boundaries) persists; report α and guidelines, and use a core 3-class scheme to reduce ambiguity.

---

## 13. One-Paragraph Pitch (for a proposal/abstract)
> Nearly all published results on code-mixed sentiment analysis for Dravidian languages are produced on a single 2019 YouTube movie-trailer corpus, in a single script, under random comment-level splits that risk near-duplicate train–test leakage, and are ranked by metrics that changed mid-stream — so the field cannot tell whether its models generalise or memorise, and its numbers are not comparable. We propose **DravidianBench-X**, the first leakage-controlled, dual-script, multi-domain, five-language (Tamil, Malayalam, Kannada, Telugu, Tulu) benchmark with a script/domain/language/time robustness suite, and **SCRIPT-D**, the first script-invariant, family-transferable, imbalance-calibrated architecture for Dravidian CMSA — combining transliteration-directed contrastive learning, LID-guided graph attention, mixture-of-family-adapters, and calibrated class-balanced heads. Evaluated across five shift axes with statistical significance and calibration reporting, SCRIPT-D is designed to establish the field's first trustworthy, robustness-aware state of the art.

---

## Appendix — Key Verified Anchors in This Document
- DravidianCodeMix: 44,020 / 19,616 / 7,671 comments; 5-class; random 90/5/5 split [V].
- Class imbalance: Tamil 56.5% positive; 2026 political 8:1 [V].
- FIRE 2020 best weighted-F1 0.65/0.74; FIRE 2021 0.711/0.804/0.630 [V].
- Tamil–Tulu 2023 best macro-F1 0.32/0.542; 2024 0.260/0.584; political-2026 best 0.3935 [V].
- 2026 DravidianLangTech has **no core SA task** [V].
- Script-invariance tooling exists (TransliCo ACL 2024; SenTAS++ IEEE 2025 for Marathi) but **not for Dravidian** [V].
- LLMs underperform fine-tuned transformers on code-mixed sentiment zero-shot; LLM augmentation helps (Zeng 2024) [V].
- Word-level LID resources for Dravidian exist (CoLI@FIRE2024, WILD@FIRE2025) [V].

Full citations and links: see `literature_review_dravidian_codemixed_sentiment_2020_2026.md`, Appendix B.
