# ML-SGCA — Executive Pitch Deck
## Matrix Language Syntax-Guided Cross-Attention for Romanized Telugu–English Code-Mixed Sentiment

**Format:** slide-by-slide  
**Date:** 2026-10-11  
**Target venues:** ACL / EMNLP / NAACL · ACM TALLIP · IEEE TASLP  
**Compute ceiling:** ≤ 2 GPU training runs (RTX 2050 4 GB or free T4)

---

## Slide 1 — Executive Title & Pitch

**Title:** *ML-SGCA: Matrix Language Syntax-Guided Cross-Attention for Romanized Telugu–English Code-Mixed Sentiment.*

**One-paragraph pitch:**
On a leakage-free Romanized Telugu–English (Tenglish) benchmark, a **0-parameter TF-IDF model (Macro-F1 0.8293) beats a fine-tuned 238M-parameter MuRIL (0.8167)**. Every standard explanation — leakage, script disparity, the Indic-pretraining paradox, subword fragmentation — is empirically dead. We are the first to explain this anomaly *structurally*: under **Myers-Scotton's Matrix Language Frame (MLF) + 4-M model**, Tenglish sentiment is carried by **morpheme-local cues** (content morphemes + Matrix-Language system morphemes) inside a **noisy, low-signal syntactic frame** — exactly what n-grams exploit and what a generic transformer over-parameterizes. We turn that theory into a **structural inductive bias**: an **MLF/4-M role-conditioned cross-attention** where content tokens attend to their matrix-language system morphemes under a soft, learned attachment prior. This is the first architecture derived from code-switching theory for sentiment classification.

**Ask / deliverable:** one LoRA training run on the frozen MuRIL backbone, plus CPU diagnostics.

---

## Slide 2 — Problem Statement & The Linear-Beats-Transformer Anomaly

**Verified leak-free Split B (CMTE-NITANP — 34,588 Romanized comments, 3-class):**

| Model | Params | Split B Macro-F1 | Acc |
|---|---|---|---|
| TF-IDF (word+char) + LogReg | 0 | **0.8293** | 0.8291 |
| TF-IDF + LinearSVC | 0 | 0.8248 | 0.8245 |
| MuRIL-base + LoRA | 592k trainable / 238M | 0.8167 | 0.8167 |
| CNN+BiLSTM (MuRIL WordPiece) | ~1.2M | 0.7945 | 0.7942 |
| CNN+BiLSTM (word) | ~1.2M | 0.7863 | 0.7858 |

**The anomaly:** a 0-parameter lexical model beats a 238M-parameter multilingual transformer by **1.26 F1**, on a leakage-free split, with near-balanced classes.

**Four hypotheses eliminated by our own experiments:**
1. **Leakage** — |SplitA−SplitB| < 0.01.
2. **Script disparity** — CMTE-NITANP is 100% Latin (0 Telugu code points).
3. **Indic-pretraining paradox** — MuRIL was pretrained on Romanized (transliterated) Indic.
4. **Subword fragmentation** — MuRIL WordPiece *helped* CNN+BiLSTM (0.7863 → 0.7945).

**Open question:** *Why* does the lexical model win — and can that reason be engineered into a model that beats it?

---

## Slide 3 — Linguistic Foundation (Myers-Scotton MLF / 4-M Model)

**MLF (Myers-Scotton):** in code-switching one language — the **Matrix Language (ML = Telugu here)** — supplies the morphosyntactic frame and the **system morphemes**; the **Embedded Language (EL = English)** contributes islands of content, inserted into the ML frame.

**4-M model (Myers-Scotton & Jake):** morphemes split into four principled types —
- **C** — content morphemes (roots: `cinema`, `bagundi`, `waste`, `keka`).
- **ES** — early system morphemes (plural, determiners).
- **LSB** — late system *bridge* morphemes (case/clitic: `-lo`, `-ki`, `-tho`, `-ga`).
- **LSO** — late system *outsider* morphemes (tense/aspect/negation/agreement: `-ledu`, `-thondi`, `-kottindi`, `kaadhu`, `kani`, `asalu`).

**System Morpheme Principle:** system morphemes come from the ML and structure the utterance — they are the *operators* that scope and flip polarity.

**Computational precedent (why this is legitimate, not hand-waving):** automatic **Matrix Language Identification** from MLF/4-M is an established NLP task (EMNLP 2024) — but only for **speech/ASR**, never as a **sentiment-attention prior**. That is our opening.

**Thesis:** sentiment is **morpheme-local**, not syntax-deep → a lexical estimator wins on small, noisy Romanized data → therefore inject the missing **structural prior** explicitly.

---

## Slide 4 — Architectural Innovation (ML-SGCA Formulation & Attention Prior)

```
  x = "aa cinema chala bagundi kaadhu bore kottindi"
        │
        ▼           4-M role tagger (automatic, § spec)
  tokens + roles ρ(i) ∈ {C, ES, LSB, LSO, O}
        │
        ▼
  ┌──────────────────────── MuRIL-base (frozen) + LoRA r=16, α=32 ────────────────────────┐
  │  H ∈ R^{L×d}   (typed self-attention with additive MLF bias M_ij^(h))                  │
  └───────────────────────────────────────────────────────────────────────────────────────┘
        │                                    │
   content set C                        system set S = ES ∪ LSB ∪ LSO
        │                                    │
        └──────────────► MLF Cross-Attention ◄┘
        c_i = Σ_{j∈S} softmax_j( (Wq h_i)·(Wk h_j)/√d + β_{ρ(i),ρ(j)} + γ A_ij ) Wv h_j
        h̃_i = LN( h_i + λ · c_i )                       # "system morphemes scope their host"
        │
        ▼
  attention-pooled content representation  →  Linear head  →  3-class sentiment
```

**Attention prior (the core equation):**
```
M_ij^(h) = b^(h)_{ρ(i), ρ(j)} + γ^(h) · A_ij
α_ij^(h) = softmax_j( (q_i^(h)·k_j^(h))/√d_h + M_ij^(h) )
b^(h)    = b + u_h v_hᵀ        # shared K×K MLF role bias + per-head rank-1, K=5
A_ij     = 1[ system morpheme j is hosted by / clitic-attached to content token i ]
```
- `b` is **initialized from an MLF prior** (system→host positive, identity ≈ 0, implausible pairs strongly negative) and then learned.
- **Soft and differentiable, never a hard mask** — so imperfect automatic tags cannot break training; hard masking is reported only as an ablation.

**Novelties:**
- **N1** First MLF/4-M-derived attention bias for any text-classification task.
- **N2** Content→system **cross-attention** that operationalizes "system morphemes scope polarity."
- **N3** A fully automatic 4-M role tagger (LID + transliteration + Telugu FST + unsupervised clitic mining) with **zero manual morpheme annotation**.

---

## Slide 5 — Why This Solves The Anomaly (Frame vs Island & System-Morpheme Salience)

- **Frame-vs-island locality.** Polarity rides on content morphemes embedded as islands; the ML frame is *low-signal and highly variable* under Romanized spelling/word order. A generic transformer spends capacity modeling a useless frame; n-grams ignore order and key on morphemes. ML-SGCA keeps the morpheme focus **and** adds the one piece n-grams lack: the **system-morpheme scope**.
- **System-morpheme salience.** LSO morphemes (`-ledu`, `kaadhu`, `kani`, `asalu`) flip/scale polarity. TF-IDF already weights them; a plain transformer disperses attention. ML-SGCA **routes attention** to the system morphemes that scope each content morpheme.
- **Falsifiable predictions (each = an experiment):**
  - P1: MLF typing of system morphemes predicts polarity shifts measureable independent of the model (feature-ablation test).
  - P2: Removing the MLF bias (random/LID-generic bias substituted) drops Macro-F1 → the *role* bias is causal, not "more attention."
  - P3: Gains concentrate on comments with negation/scope operators (error-subset analysis).

---

## Slide 6 — Novelty Landscape & Reviewer-Proof Defense

| Prior art | What it does | How ML-SGCA differs |
|---|---|---|
| Automatic MLID (EMNLP 2024) | MLF/4-M for **speech** (ASR) | First use as a **sentiment attention prior** for **text** |
| SenTAS++ (IEEE 2025) | language-**tag graph** attention (Marathi) | MLF **role/scope** bias, not LID adjacency |
| MultiSwitchNet (KBS 2024) | socio-**pragmatic** framework (Dravidian) | ML **grammatical** frame, not pragmatics |
| CharBERT / CharacterBERT | char+subword **fusion** | no fusion — a **structural attention bias** |
| CMSAOne (FIRE 2020) | TF-IDF+ELMo+fastText meta-embedding | no lexical fusion; theory-grounded bias |
| UDA / VAT / R-Drop / SCR | perturbation **consistency** | no augmentation, no consistency loss |

**Killer objections → answers:**
- *"MLF is a 1990s sociolinguistic theory."* → We don't claim it's true of every utterance; we claim it is a **measurable, falsifiable structural prior**, with computational precedent (MLID), and we ablate it.
- *"This is just another attention variant."* → The **role-vs-random-vs-LID** bias ablation isolates the MLF contribution.
- *"Only one dataset / one seed."* → ≥5 seeds + 95% CIs + paired significance; CMTET cross-corpus + a second Dravidian language.

---

## Slide 7 — Experimental Pipeline & Resource Budget (≤ 2 GPU runs on RTX 2050)

| Stage | What | Hardware | Time |
|---|---|---|---|
| S0 | 4-M role tagger (LID + IndicXlit + Telugu FST + clitic mining) | CPU | ~30 min |
| S1 | **Run 1 (GATE):** full MuRIL fine-tune (decides if the anomaly survives) | GPU | ~60 min |
| S2 | **Run 2:** ML-SGCA (frozen MuRIL + LoRA + MLF cross-attention) | GPU | ~50 min |
| S3 | Ablations (bias: MLF / random / LID-generic / none) | GPU (reuse) or CPU | ~1 h |
| S4 | Diagnostic ablations, CMTET cross-corpus, tagger sensitivity | CPU/inference | ~1–2 h |

**Measured VRAM envelope** (real MuRIL+LoRA run = 1.58 GB / 3.87 GB at max_len 64, batch 16×2, bf16). ML-SGCA adds role embeddings + a K×K bias table (~145 params) + one cross-attention block → memory-neutral. Config: len 64, batch 16–24, bf16, LoRA r16 α32, ~8 epochs.

---

## Slide 8 — Expected Margins & Target Venues

**Honest margin expectation (no fabrication):**
- Target: ML-SGCA Macro-F1 **≥ 0.845** on Split B (≥ +0.015 over LogReg 0.8293), with a **mechanism ablation** proving the MLF bias is the cause.
- Realistic distribution: decisive (>+0.015) ≈ 40%; modest (+0.005–0.015) ≈ 40%; tie/loss ≈ 20%. The decisive outcome is *conditional* on the gate run (S1) surviving — i.e., on full fine-tuning not already clearing 0.83.

**Positioning if the margin is modest:** publish as a **structural-bias / analysis** contribution ("what code-switching theory predicts about neural sentiment") — honest, novel, still first-of-its-kind.

**Venues:**
- **Primary:** ACL / EMNLP / NAACL main or Findings (if generality across ≥2 datasets + decisive margin).
- **Strong fit:** **ACM TALLIP**; **IEEE TASLP**.
- **Safety net:** DravidianLangTech / VarDial workshop (fast, high-fit) → extend to journal.
- **Cost note:** workshops are near-free; ACM/IEEE journals are subscription or gold-OA (APC applies — verify the current fee before budgeting).

**What we explicitly do NOT claim:** SOTA on all datasets, or a gain that survives beyond our leak-free protocol without the gate experiment.
