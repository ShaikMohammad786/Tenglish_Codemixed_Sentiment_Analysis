# Code-Mixed Sentiment Analysis in Dravidian Languages (2020–2026)
## A Verified, Evidence-Anchored Systematic Literature Review

**Review date:** 28 September 2026
**Scope:** Sentiment analysis (SA) of code-mixed / code-switched social-media text in the Dravidian language family (Tamil, Malayalam, Kannada, Telugu, Tulu) mixed with English, 2020 → 2026.
**Method:** Structured search of ACL Anthology (all DravidianLangTech volumes 2021–2026, FIRE working notes 2020–2022), arXiv, Springer, IEEE Xplore, ScienceDirect, and ACM DL, plus direct verification of primary shared-task overview papers.
**Verification policy:** Every quantitative claim below is tagged **[V]** = verified against a primary source during this review, or **[P]** = plausible/commonly reported but *not* independently verified here. Numbers from different papers are **never** placed on one leaderboard (see §7.4).

> This review also audits three prior AI-generated documents on the same topic (see Appendix A). Two contain material errors; one is largely accurate. Those corrections are integrated throughout.

---

## Abstract

Between 2020 and 2026, sentiment analysis for Dravidian code-mixed text grew from a single FIRE pilot into a sustained programme organised around a recurring shared-task series (FIRE 2020–2022; DravidianLangTech 2021–2026) and a growing independent journal literature. This review consolidates the verified evidence across roughly 70 primary sources and shows that: (i) the field's apparent "collapse in accuracy" from ~0.71 to ~0.26 on Tamil is **an artefact of switching the primary metric from weighted-F1 to macro-F1 and changing the dataset**, not a true regression [V]; (ii) the strongest practical baselines are fine-tuned multilingual transformers (XLM-R, MuRIL, IndicBERT, mBERT), with classical models still competitive on the smallest resources [V]; (iii) the negentropy of the field is low — a small set of datasets (dominated by DravidianCodeMix) and a small set of architectures (transformer fine-tuning) are recycled, while the genuinely hard problems (script/transliteration robustness, cross-domain and intra-Dravidian transfer, sarcasm/implicit sentiment, evaluation validity/leakage, calibration, explainability) remain open [V]. The single most consequential and least-addressed gap is **generalisation validity under script, domain, and language shift combined with leakage-contaminated evaluation**. This review closes by specifying that gap precisely; the companion document (`research_gap_and_publishable_architecture.md`) proposes a publishable architecture and benchmark that attack it.

---

## 1. Introduction

India's online discourse is overwhelmingly multilingual, and users routinely blend English with a native language inside a single sentence, clause, or word — **code-mixing** (CM) / **code-switching** (CS). In the Dravidian languages of South India, a second layer of complexity is added: because Dravidian scripts (Tamil, Malayalam, Kannada, Telugu abugidas; Tulu's dormant Tigalari) are inconvenient on QWERTY keyboards, users frequently **transliterate** their language into Roman script. The result mixes two languages *and* two scripts simultaneously (`"enaku ithu super ah iruku"`, `"bhayankara mass da"`), which is precisely where monolingual and even multilingual NLP systems degrade.

Before 2020 there was no shared benchmark for Dravidian code-mixed sentiment. The **Dravidian-CodeMix** track at FIRE 2020 changed that, and a research programme has run continuously since. This review answers four questions:

- **RQ1 — Scope & evolution:** How has the task been defined, scaled, and extended (new languages, modalities, label schemes) across 2020–2026?
- **RQ2 — Methods:** Which modelling paradigms have been applied, and where has the frontier actually moved?
- **RQ3 — Data:** What resources exist per language pair, at what scale, with what labels and licences?
- **RQ4 — Open problems:** Which limitations recur, and which constitute a defensible novel contribution?

**A note on "state of the art."** The most common misinterpretation in this literature — reproduced in one of the seed documents (Appendix A) — is to read the fall from Tamil weighted-F1 ≈ 0.71 (2021) to macro-F1 ≈ 0.26 (2024) as a collapse in model quality. It is not. The 2020–2022 editions ranked systems by **weighted** average F1 on a majority-dominated 5-class scheme; the 2023–2026 editions switched to **macro** average F1 on a re-annotated 4-class scheme [V]. Macro-F1 on an imbalanced 4-class problem is intrinsically far lower. Cross-era comparisons are therefore invalid, and this review is careful to keep metric regimes separate (§7.4).

---

## 2. Background

### 2.1 The language family and its scripts
Tamil, Telugu, Kannada, and Malayalam are the four major literary Dravidian languages (all official/scheduled in India; Tamil also official in Sri Lanka and Singapore). All four are **agglutinative** (long suffix chains) and all four use distinct **abugida** scripts. Tulu (~2.5M speakers, coastal Karnataka/Kerala) retains ancient Dravidian features and originally used the **Tigalari** script, now largely displaced by Kannada script; it entered computational SA only in 2023 [V].

### 2.2 Typology of code-mixing in the reviewed corpora
All reviewed datasets contain the three canonical levels, often in Roman script:

| Type | Description | Corrupted example (from FIRE 2021 paper) |
|---|---|---|
| Inter-sentential | Whole clauses/sentences alternate | `"Omg .. use head phones. Enna bgm da saami .."` |
| Intra-sentential | Switch within a clause | `"I think sivakarthickku hero getup set aagala."` |
| Tag-switching / script-mixing | Inserted markers/words; also Roman↔native script mixing | `"Yaru viswasam teaser ku marana waiting like pannunga"` |
| Intra-word | Mixing at morpheme level | Kannada/Tamil stems + English affixes |

[V — verbatim examples reproduced from the Dravidian-CodeMix 2021 findings.]

### 2.3 Levels of SA
Document-, sentence-, and aspect-level SA all exist in general NLP; the Dravidian code-mixed literature operates almost entirely at the **sentence/comment level**. Aspect-level (ABSA) and document-level treatments are rare (§8.6) [V].

### 2.4 Label schemes (and why they matter)
- **FIRE 2020–2022 scheme (5 classes):** Positive, Negative, Mixed_feelings, Neutral/unknown_state, Not-in-intended-language [V].
- **DravidianLangTech 2023–2025 scheme (4 classes):** Positive, Negative, Neutral (or unknown), Mixed Feeling; the "not-in-language" category was dropped for Tamil/Tulu [V — 2023 findings enumerate exactly four].
- **2026 political scheme (7 classes):** Substantiated, Sarcastic, Opinionated, Positive, Negative, Neutral, None of the above [V].

Because these schemes differ, and because the "anonymous/mixed/neutral" boundaries are linguistically fuzzy, cross-edition comparison is compromised *by construction*. This is a recurring theme (§9.2).

---

## 3. Timeline of Shared Tasks and Benchmarks (2020–2026)

| Year | Venue / track | Languages | Structural features | Best reported result |
|---|---|---|---|---|
| **2020** | FIRE — Dravidian-CodeMix | Tamil-En, Malayalam-En | First Dravidian CM-SA benchmark | Weighted-F1 **0.65** (Tamil), **0.74** (Malayalam) [V] |
| **2021** | FIRE — DravidianCodeMix | + Kannada-En | Full DravidianCodeMix released; 120 registered teams (22/17/15 systems) | Weighted-F1 **0.711 / 0.804 / 0.630** [V] |
| **2021** | 1st DravidianLangTech workshop (co-located) | Tamil, Malayalam, Kannada | Companion academic venue launched | — |
| **2022** | FIRE 2022 — DravidianCodeMix | Tamil, Malayalam, Kannada | Task rebranded "Sentiment Analysis **and Homophobia Detection**" | — [V] |
| **2022** | DravidianLangTech@ACL | Tamil (+ Telugu/Malayalam adjacent) | **Multimodal** SA (video) + troll-meme classification; separate Tamil emotion task | Multimodal SAS: only 1 team submitted, F1 **0.24** [V] |
| **2023** | DravidianLangTech@RANLP | Tamil, **Tulu** (new) | First public Tulu SA task; 57 registered teams, 27 systems | Macro-F1 **0.32** (Tamil), **0.542** (Tulu) [V] |
| **2024** | DravidianLangTech@EACL | Tamil, Tulu | 2nd Tamil–Tulu task; 64 registered (19/17 systems); HOLD-Telugu hate-speech in parallel | Macro-F1 **0.260** (Tamil), **0.584** (Tulu) [V] |
| **2025** | DravidianLangTech@NAACL | Tamil, Tulu (+ Multimodal Tamil/Malayalam/Telugu) | 3rd Tamil–Tulu task, 4-class; MSMDA-DL multimodal hate speech; AI-generated-review detection (Malayalam/Tamil) | Tamil (Tulu track) systems ~0.44–0.54 macro-F1 [V, per-team] |
| **2026** | DravidianLangTech@ACL | Tamil, Tulu, Telugu, Malayalam | **No core Tamil–Tulu SA task.** Political multiclass SA (7-class, 5,440 tweets), Hope-Speech (Tulu), Political Meme (Tamil/Malayalam), Abusive-Tamil-targeting-women, Prompt Recovery (Telugu), Depression-from-speech, Dialect speech | Political SA best Macro-F1 **0.3935** [V] |

**Three structural observations.**
1. The *core recurring* sentiment task narrowed after 2021 to a **Tamil–Tulu** pairing; Malayalam and Kannada sentiment migrated into occasional multimodal tasks or independent papers; **Telugu was structurally excluded** from the core sentiment line (appearing only in adjacent hate-speech tasks) [V].
2. In **2026 the core sentiment task disappeared entirely**, replaced by domain-specific (political) and adjacent tasks — a signal that the generic 5-class line has plateaued [V].
3. The modelling stack shifted: 2020–2021 classical ML/early transformers → 2022–2024 transformer fine-tuning → 2025–2026 **PEFT (LoRA) + LLM-based synthetic augmentation** [V].

---

## 4. Datasets and Resources

### 4.1 Core corpora (verified sizes)

| Dataset | Pair(s) | Size | Labels | Source | Licence |
|---|---|---|---|---|---|
| **TamilMixSentiment** (Chakravarthi 2020) | Tamil-En | **15,744** comments | 5-class | YouTube (movie trailers) | GitHub / CC-BY-NC [P licence] |
| **MalayalamMixSentiment** (Chakravarthi 2020) | Malayalam-En | **6,739** comments (FIRE edition used ~6,000) | 5-class | YouTube | GitHub [P licence] |
| **KanCMD** (Hande 2020, PEOPLES) | Kannada-En | **7,671** comments | Sentiment + Offensive (multi-task) | YouTube | GitHub |
| **DravidianCodeMix** (Chakravarthi 2022, LRE) | Tamil/Malayalam/Kannada-En | **44,020 / 19,616 / 7,671** comments | 5-class sentiment + offensive | YouTube | GitHub/Zenodo **(CC-BY default)** |
| **Tamil–Tulu 2023 corpus** | Tamil-En, Tulu-En | Tamil **33,989 train + 3,786 dev**; Tulu **6,458 + 781** | 4-class | YouTube | Task release |
| **TamilPoliSent 2026** | Tamil | **5,440** tweets | 7-class political | X/Twitter | Task release |
| **TamilEmo** (2022) | Tamil | **>42,000** comments | 3 / 7 / 31 emotion classes | YouTube | GitHub |
| **MADTRAS** | Tamil | **4,000** reviews, **7,507** aspect instances, 9 aspects | ABSA | Movie reviews (survey+IMDb+YouTube) | Mendeley |
| **Sarcasm (Tamil, Malayalam)** (2025) | Tamil-En, Malayalam-En | Sarcasm-labelled YouTube subset | Sarcasm binary | YouTube | Springer SNAM |

**[V]** for all sizes above except the noted licence caveats.

### 4.2 Class distribution (the imbalance problem, quantified)

From the DravidianCodeMix-2021 corpus statistics table [V]:

| Class | Tamil | Malayalam | Kannada |
|---|---|---|---|
| Negative | 5,228 (11.9%) | 2,600 (13.3%) | 1,484 (19.3%) |
| Not-in-language | 2,087 (4.7%) | 1,445 (7.4%) | 1,136 (14.8%) |
| Neutral | 6,904 (15.7%) | 6,502 (33.1%) | 842 (11.0%) |
| Mixed feelings | 4,928 (11.2%) | 1,162 (5.9%) | 691 (9.0%) |
| **Positive** | **24,873 (56.5%)** | **7,907 (40.3%)** | **3,518 (45.9%)** |
| **Total** | **44,020** | **19,616** | **7,671** |

Train/dev/test split = **90/5/5** for every language [V]. In the 2023 Tamil–Tulu corpus, Tamil's Positive class alone was 20,070 of ~37,775 train+dev instances, and Tulu's Positive (3,118) and Neutral (1,719) dominated [V]. In the 2026 political task the imbalance is explicitly reported at **8:1** [V]. Severe, persistent imbalance is thus a *structural property* of every Dravidian CM-SA dataset, and it is the reason macro-F1 became the standard metric after 2022.

### 4.3 Auxiliary resources (important, under-used)
- **Word-level language identification (LID):** DravidianCodeMix carries word-level tags; dedicated shared tasks **CoLI@FIRE2024** and **WILD@FIRE2025** target word-level LID in Dravidian CM text [V]. Yet LID is usually *preprocessing*, not a modelled signal (§5.3).
- **Offensive-language annotations:** bundled with DravidianCodeMix and KanCMD — enabling multi-task learning that remains largely unexploited.
- **Transliteration tooling:** AI4Bharat's **IndicXlit** and **IndicLID**, plus Google `googletrans`-based LID used in the Cambridge NLP paper [V].

### 4.4 Critical data caveats
1. **Domain homogeneity.** Nearly every corpus is **YouTube movie-trailer comments** from ~2019, encoding actor/film/platform-specific vocabulary [V]. Cross-domain validity is essentially untested.
2. **Random comment-level splits.** DravidianCodeMix uses a 90/5/5 *random* split [V]. Because comments on the same trailer are near-duplicates (`"super"`, `"mass"`, blocking-copy spam), random splits risk **train/test leakage** and inflate accuracy. This is not hypothetical: near-duplicate reuse is a documented benchmark-leakage mechanism [V, general leakage literature]. Hardly any reviewed paper uses video-level, temporal, or topic-level splits.
3. **Single-script bias.** Corpora are predominantly Roman-script; models are rarely evaluated on the native-script or mixed-script counterparts of the same content, even though the corpora explicitly contain both [V].
4. **Annotation scheme drift.** The 5→4→7 class transitions across editions break longitudinal comparison [V].

---

## 5. Methodological Evolution

### 5.1 Era summary

| Period | Dominant paradigm | Representative techniques |
|---|---|---|
| 2020–2021 | Classical ML + early transformers | TF-IDF/char-n-gram + SVM/NB/LR/RF; ensembles; ULMFiT; XLM-R fine-tuning; meta-embeddings + GRU |
| 2021–2023 | Deep learning | Bi-LSTM, Bi-GRU, CNN+Bi-LSTM, hierarchical/ multi-head attention, syllable+word embeddings (Telugu), GCN+multi-head attention |
| 2022–2025 | Transformer fine-tuning (dominant) | mBERT, XLM-R, IndicBERT, MuRIL, RoBERTa, Distil-mBERT; ensembles (soft voting); transliteration/translation variants |
| 2023–2026 | LLM-assisted & PEFT | Zero/few-shot GPT prompting (often underperforming); LLM synthetic augmentation (Gemini 2.5 Flash); LoRA/adapters; hybrid encoder–decoder |

### 5.2 What actually won (verified)
- **FIRE 2021** Tamil #1 (CIA_NITT) used **XLM-R with a paraphrase objective**; Malayalam #1 (ZYBank-AI) used **XLM-R + added self-attention over the 12 hidden layers** + two-stage domain-adaptive pretraining; **Kannada #1 (SSNCSE_NLP) used classical ML** (LR/SVM on TF-IDF), beating all transformer entries [V]. Lesson: on the smallest, noishest resource, classical models remain competitive.
- **FIRE 2020** top runs (weighted-F1 0.65/0.74) likewise relied on transformer/character-subword features [V].
- **2023 Tamil–Tulu** #1 (DeepBlueAI) = fine-tuned **XLM-RoBERTa** with mixed-language data proportions + cross-validation; Tamil #2 used **Tamil-sentiment-BERT + oversampling**; **ELMo** and **fastText+LinearSVC** also placed [V].
- **2024** the CUET system (XLM-R/mBERT) reached macro-F1 **0.258 (Tamil) / 0.468 (Tulu)**, ranking 2nd/5th [V].
- **2025** a Support-Vector-Machine system reached macro-F1 **0.438 (Tamil) / 0.54 (Tulu)** [V], again showing how weak the "deep" frontier is on Tamil.
- **2026 political SA:** best macro-F1 **0.3935** across 22 teams; leading systems used **LoRA-tuned XLM-R** and **weighted transformer ensembles**, plus **Gemini-2.5-Flash augmentation** (CHMOD_777 expanded training data using LLM-generated examples under an 8:1 imbalance) [V].

### 5.3 Handling the script/transliteration problem
Four strategies recur [V from Doc C, confirmed by the Cambridge NLP paper's three RQ angles]:
1. **Keep original noise** (preserves switch points; harms tokenisation).
2. **Transliterate Roman→native** (increases lexical overlap with native-script pretraining; may erase dialect/identity signal).
3. **Translate to English** (normalises; *destroys* code-switch structure and pragmatics).
4. **Multi-view** (concatenate original + transliterated + translated; costly and leakage-prone).

The **word-level-LID-as-feature** approach (Chanda et al., *Natural Language Processing*, 2024/25) shows LID improves Dravidian CM-SA, and explicitly frames the question *"when should a dataset be called code-mixed?"* around script mixture — but LID is still used as a *feature*, not as an architectural prior [V].

### 5.4 Graph, attention, and hybrid approaches
- **GCN with multi-headed attention** (Dowlagar & Mamidi, DravidianLangTech 2021) is the field's only well-known graph attempt for Dravidian CM-SA — a proof of concept, not a sustained line [V].
- **MultiSwitchNet** (Jothi Prakash V. et al., *Knowledge-Based Systems* 2024) proposes a **socio-pragmatic** framework for Dravidian-English code-switched SA, adding pragmatic/switch features [V]. (Note: this is the strongest *architectural* prior art in the Dravidian space and must be cited as such.)
- **CDMG-BERT** (domain-adaptive multitask **graph-enhanced BERT**) exists for **Hinglish**, not Dravidian [V] — a clear porting gap.
- **HLM** (EMNLP 2025) combines an mBERT encoder with a lightweight decoder (<3B params) and matches fine-tuned LLMs on code-mixed sentiment, arguing efficiency [V].

### 5.5 The LLM era — what the evidence actually says
Two distinct threads, with a consistent conclusion:

1. **LLMs as zero/few-shot classifiers.** Across studies, general LLMs **underperform fine-tuned multilingual transformers** on code-mixed sentiment, especially on minority classes; zero-shot prompting of complex strategies often loses to simple prompts, and LLMs systematically degrade on code-mixed inputs spanning multiple language families [V — multiple studies]. This holds for Dravidian and for code-mixed SA generally.
2. **LLMs as data-augmentation engines.** More promising: few-shot LLM generation of synthetic CM examples to rebalance classes. Zeng et al. (arXiv:2411.00691) report **+9.32% F1 on Spanish–English** and also work on **Malayalam–English** [V]; 2026 DravidianLangTech systems use Gemini-2.5-Flash augmentation [V]. Risk: synthetic-data distribution drift and contamination of test content.
3. **Parameter-efficient adaptation.** LoRA/adapters are now the default way to fit large multilingual models to small Dravidian sets — but the literature has **no systematic comparison of LoRA vs adapters vs prefix/prompt tuning vs full fine-tuning on a fixed Dravidian CM-SA split** [V].

### 5.6 Transliteration-contrastive and script-invariance priors exist — but not for Dravidian
- **TransliCo** (ACL 2024) uses contrastive learning to break the **script barrier** in multilingual PLMs [V].
- **SenTAS++** (IEEE 2025) unifies **transliteration + token-level LID + cross-view contrastive fusion + language-tag graph attention for code-mixed Marathi** [V].
- **AdapterFusion-based multi-task learning** for code-mixed/code-switched text (EAAI 2023) is described by its authors as the *first* language-based AdapterFusion [V].

These establish that the *techniques* are publishable and effective — and simultaneously establish that **no one has assembled them into a coherent, Dravidian-family, script-robust, imbalance-calibrated, leakage-free system.** That intersection is the anchor of the proposed research (§10).

---

## 6. Language-by-Language Assessment

| Pair | Largest resource | Shared-task continuity | Reported performance regime | Maturity |
|---|---|---|---|---|
| **Tamil-En** | 44,020 (DravidianCodeMix) + 33,989 (Tamil–Tulu) | Continuous 2020–2025 | Weighted-F1 0.65–0.711 (2020–21); macro-F1 0.26–0.44 (2023–25) | Most studied; widest task coverage (SA, sarcasm, hate, emotion, political) |
| **Malayalam-En** | 19,616 | Frequent, via adjacent/multimodal tasks | Weighted-F1 0.74–0.804 | Mature; strong ABSA/multimodal extensions |
| **Kannada-En** | 7,671 (= KanCMD) | **Discontinued after 2022** | Weighted-F1 0.630 | Stalled; resource not renewed |
| **Telugu-En** | Fragmented (~19–20k in individual studies) | **Absent from core SA task** | Accuracy ~0.80 (3-class, unnormalised protocols) | Least standardised despite output |
| **Tulu-En** | 6,458 train | Continuous 2023–2026 | Macro-F1 0.20–0.584 | Newest; most volatile |
| Emerging (Kodava, Beary, Tulu dialects) | — | — | — | No benchmark |

**[V]** for all rows.

Key asymmetry: **Tamil and Malayalam carry nearly all the international benchmark attention; Kannada froze in 2021; Telugu never had a comparable SA leaderboard; Tulu is tiny.** A cross-language model that transfers *within* the family is therefore both scientifically attractive and practically absent.

---

## 7. Cross-Cutting Themes and Evidence Synthesis

### 7.1 Sentiment/offence overlap
A large share of "sentiment" papers are effectively **joint sentiment + offensive/hate** studies, because DravidianCodeMix, KanCMD, and later HOLD-Telugu were built as multi-task resources [V]. Multi-task learning is repeatedly *suggested* but rarely done systematically for Dravidian CM-SA.

### 7.2 Sarcasm, irony, implicit sentiment
Sarcasm is flagged as a dominant error source in shared-task retrospectives; a dedicated **Tamil+Malayalam sarcasm dataset** (SNAM 2025) with **SHAP** explainability now exists [V], and FIRE 2023 ran a "Sarcasm Identification of Dravidian Code-Mixed" task [V]. In 2026, sarcasm became an explicit *class* in the political scheme [V]. Yet no sarcasm/negation-aware *architecture* for Dravidian CM-SA has been proposed.

### 7.3 Multimodality
Introduced in 2022 (video SA + troll memes); the 2022 multimodal SA task drew **only one team (F1 0.24)** [V]. By 2025 the multimodal line covers Tamil/Malayalam/Telugu **hate speech with audio**, but remains small and below text-only reliability [V].

### 7.4 Metric incomparability (a first-class problem)
- **Weighted-F1** (2020–22) is dominated by the Positive majority class → inflated.
- **Macro-F1** (2023–26) weights all classes equally → much lower, and highly sensitive to rare-class behaviour.
- Test sets are small (2021: Tamil test = 4,402; Kannada test = 768) [V]; most papers report no confidence intervals or significance tests.
- Accuracy is sometimes reported on ~57% majority-class data, where a trivial majority predictor scores ~0.57 [V].

**Consequence:** the field cannot currently demonstrate cumulative progress, because it never fixed a benchmark × metric × split regime. This is itself a publishable finding.

### 7.5 Low-resource/Telugu specifics
Telugu work uses syllable+word embeddings, unsupervised normalisation + MLP (~80% accuracy, 3-class), and transformer classifiers, but across **non-overlapping corpora** [V]. Results are not comparable.

### 7.6 Class imbalance
Every corpus is skewed (§4.2). Mitigations tried (SMOTE, oversampling, class weights, focal loss) vary widely; a **Code-Mixing-Index-weighted focal loss** was proposed for Dravidian *offence* detection [V] but is not standard in SA; no consensus best practice emerged.

---

## 8. Research Gaps

Ordered roughly by (importance × under-addressed), with the *crucial* gap isolated in §8.1.

1. **Generalisation validity + evaluation leakage (THE crucial gap).** No reviewed work jointly tests script shift + domain shift + language shift, and nearly all benchmarks use random comment-level splits that risk near-duplicate leakage. High impact; low novelty of "idea" but high novelty of "rigorous execution."
2. **Script / transliteration robustness.** Models are trained and tested on essentially one script; the same content in the other script is untested. No script-invariant model exists for Dravidian CM-SA.
3. **Intra-Dravidian transfer.** Transfer *within* the family (Tamil→Tulu, Telugu→Kannada) is unexplored despite shared typology; the family structure is a natural, unused prior.
4. **No unified multi-language benchmark.** DravidianCodeMix is the only 3-language resource, is Roman-script-biased, randomly split, 2021-era, and 5-class. A versioned, de-duplicated, multi-script, cross-lingual benchmark does not exist.
5. **Sarcasm / implicit sentiment / negation** unmodelled architecturally.
6. **Sub-optimal use of existing supervision:** word-level LID, offensive labels, and code-mix structure are available but under-exploited.
7. **ABSA for code-mixed Dravidian** ≈ non-existent (MADTRAS is Tamil-monolingual movie reviews).
8. **Calibration & uncertainty** never reported, despite high-stakes moderation use.
9. **Explainability** minimal (one SHAP study).
10. **Fairness/dialect robustness** unmodelled; datasets treat each language as monolithic.
11. **Deployment realities** (latency, drift, continual learning) absent.
12. **Multimodality** immature.
13. **Federated/privacy-preserving CM-SA** essentially absent (a recent FedPerX adapter+DP framework exists for *general multilingual* sentiment [V], not Dravidian).
14. **Evaluation-metric/reporting inconsistency** (§7.4).

---

## 9. Future Directions (field-level)
(a) A **unified, versioned, leakage-free, multi-script benchmark** across Tamil/Malayalam/Kannada/Telugu/Tulu with fixed label mapping and official video/temporal splits; (b) **script- and transliteration-robust** modelling; (c) **intra-Dravidian transfer** exploiting family structure; (d) **sarcasm/negation-aware** heads; (e) **systematic PEFT** comparison (LoRA vs adapters vs prompts); (f) **calibration + explainability** as default reporting; (g) **deployment-oriented** evaluation (drift, latency). The companion document designs items (a)–(c) into a single publishable programme.

---

## 10. Conclusion
Dravidian code-mixed sentiment analysis matured from a 2020 pilot into a multi-language, multi-task programme, but its centre of gravity (DravidianCodeMix + transformer fine-tuning, ranked by an incomparable metric on randomly split movie-trailer comments) has barely moved since 2021. The frontier questions — does a model survive a change of **script**, **domain**, **dialect**, or **neighbouring Dravidian language**, and are current numbers even trustworthy given leakage-prone splits? — remain almost entirely open. Those questions define the field's best-available novel contribution.

---

## Appendix A — Audit of the Three Seed Documents

| Claim / aspect | Seed Doc | Verdict |
|---|---|---|
| "Tamil: Macro F1 ≈ 0.26 (best ensemble)" attributed to TamilMixSentiment/FIRE 2020 | A (`deep-research-report.md`) | **WRONG attribution.** 0.26 is the 2024 Tamil–Tulu *macro*-F1 best. FIRE 2020 Tamil best was 0.65 *weighted* F1. Metric regimes conflated. |
| FIRE best weighted-F1 ≈0.711/0.804/0.630 | A | **Correct** [V] but it is the **2021** DravidianCodeMix task, not 2020. |
| "Eusha et al. (2024) … macro F1 0.227 on Tamil, rank 3" | A | **Unverified/likely misattributed.** The verified 2024 CUET transformer result is **0.258 Tamil / 0.468 Tulu** (2nd/5th) [V]. |
| MalayalamMixSentiment ~6,739 | A | Correct [V]. |
| KanCMD 7,671, multi-task | A | Correct [V]. |
| DravidianCodeMix "~44k/20k/7k, CC-BY" | A | Correct on size; **CC-BY** is Zenodo's default and plausible, not explicitly confirmed [P]. |
| Full 2020–2026 shared-task lineage table | B (HTML) | **Largely accurate and the best of the three** [V]. |
| "0.82 macro-F1 Tamil caste/migration hate-speech (EACL 2024)" | B | **Needs caution.** Verified independent caste/migration dataset peak macro-F1 = **0.73** [V]; a 2025 shared-task system hit **0.80** [V]. 0.82 unverified. |
| "TamilEmo ~42,000, 31 emotion classes" | B | Correct (31-class is one of 3/7/31 groupings) [V]. |
| "DravidianLangTech 2026: TamilPoliSent 7-class; Tulu hope speech" | B | Correct [V]. **Omission:** there is **no core sentiment task** in 2026 [V]. |
| "1,000–1,300 Malayalam test (4,851/540/1,348)" | B | **Unverified** [P]. |
| DravidianCodeMix "41,933 Tamil / 18,171 Malayalam / 6,535 Kannada sentiment comments" | C (PDF) | **Not the canonical figures.** Verified corpus = **44,020 / 19,616 / 7,671** [V]. C's numbers appear to be a subset/provenance variant; must be re-checked before citation. |
| "ELMo 0.2877 Tamil / 0.5133 Tulu (2023)" | C | **Not in the verified 2023 rank list** (verified: AlphaBrains ELMo 0.215/0.522; MUCS fastText 0.141/0.204) [V]. Re-check source. |
| 2024 best macro-F1 0.260/0.584 | C | Correct [V]. |
| Proposed framework: heterogeneous graph transformer + multitask + contrastive | C | **Defensible but not novel-and-feasible at once**; a full heterogeneous graph + 6 heads + federated evaluation is a high-risk, broad programme. Superseded by the focused design in the companion document. |
| Gap list (15 items) | C | **Strong and substantially correct**, especially #15 (leakage) and #4 (transliteration uncertainty) [V]. |
| "Top systems used meta-embeddings + transformer/GRU hybrids (2020)" | B | Consistent with FIRE 2020 literature [P/V]. |

**Bottom line on the seeds:** Doc B (HTML) is the most accurate; Doc A is thin and has at least two erroneous attributions; Doc C is the most useful structurally but contains incorrect dataset counts and one unverifiable result table, and its proposed framework is over-broad. None of the three exploits the **evaluation-validity/leakage** gap as the centrepiece, and none proposes a **script-invariant, family-transfer** architecture grounded in the existing (TransliCo / SenTAS++ / AdapterFusion) literature.

---

## Appendix B — Verified Reference List (with links)

1. Chakravarthi et al. (2020). *Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text.* SLTU. https://aclanthology.org/2020.sltu-1.28/
2. Chakravarthi et al. (2020). *A Sentiment Analysis Dataset for Code-Mixed Malayalam-English.* SLTU. https://aclanthology.org/2020.sltu-1.25/ · arXiv:2006.00210
3. Hande, Priyadharshini & Chakravarthi (2020). *KanCMD.* PEOPLES. https://aclanthology.org/2020.peoples-1.6/
4. Chakravarthi et al. (2022). *DravidianCodeMix.* Language Resources & Evaluation 56:765–806. https://link.springer.com/article/10.1007/s10579-022-09583-7 · Zenodo 4750858 · arXiv:2106.09460
5. Chakravarthi et al. (2020). *Overview of the Track on Sentiment Analysis for Dravidian Languages in Code-mixed Text.* FIRE. https://dl.acm.org/doi/10.1145/3441501.3441515
6. Chakravarthi et al. (2021). *Findings of the Sentiment Analysis of Dravidian Languages in Code-Mixed Text.* FIRE. arXiv:2111.09811
7. Priyadharshini et al. (2021). *Overview of the DravidianCodeMix 2021 Shared Task.* https://dl.acm.org/doi/10.1145/3503162.3503177
8. Hegde et al. (2023). *Findings of the Shared Task on Sentiment Analysis in Tamil and Tulu.* https://aclanthology.org/2023.dravidianlangtech-1.9/
9. Sambath Kumar et al. (2024). *Overview of Second Shared Task on SA in Code-mixed Tamil and Tulu.* https://aclanthology.org/2024.dravidianlangtech-1.10/
10. Durairaj Thenmozhi et al. (2025). *Overview of the Shared Task on SA in Tamil and Tulu.* https://aclanthology.org/2025.dravidianlangtech-1.124/
11. Vegupatti et al. (2026). *TamilPoliSent 2026.* https://aclanthology.org/2026.dravidianlangtech-1.15/
12. DravidianLangTech-2026 Shared Tasks. https://sites.google.com/view/dravidianlangtech-2026/shared-tasks-2026
13. Premjith et al. (2022). *Findings: Multimodal SA and Troll Meme Classification.* https://aclanthology.org/2022.dravidianlangtech-1.39/
14. Dowlagar & Mamidi (2021). *GCN with Multi-headed Attention for Code-Mixed SA.* https://aclanthology.org/2021.dravidianlangtech-1.8/
15. Chanda et al. (2024/25). *Sentiment analysis of code-mixed Dravidian languages leveraging pretrained model and word-level language tag.* Natural Language Processing (Cambridge). https://www.cambridge.org/core/journals/natural-language-processing/article/024F7BA5EE3F5F29CDB20EC9AD79CCE5
16. Jothi Prakash V. et al. (2024). *A novel socio-pragmatic framework for SA in Dravidian-English code-switched texts (MultiSwitchNet).* Knowledge-Based Systems. https://doi.org/10.1016/j.knosys.2024.112248
17. Chakravarthi (2025). *Sarcasm Detection in Tamil and Malayalam YouTube Comments.* Social Network Analysis & Mining. https://link.springer.com/article/10.1007/s13278-025-01486-z
18. Chakravarthi et al. (2025). *Detecting caste and migration hate speech in low-resource Tamil.* LRE. https://doi.org/10.1007/s10579-025-09848-x
19. Vasantharajan et al. (2022). *TamilEmo: Fine-grained Emotion Detection Dataset for Tamil.* arXiv:2202.04725
20. *MADTRAS: Dataset for ABSA of Movie Reviews in Tamil.* Mendeley. https://data.mendeley.com/datasets/p59cfx4vx6
21. Zeng et al. (2024). *Leveraging LLMs for Code-Mixed Data Augmentation in Sentiment Analysis.* SICON; arXiv:2411.00691. https://aclanthology.org/2024.sicon-1.6/
22. Balaga & Karthik (2025). *Power doesn't reside in size: A Low-Parameter Hybrid Language Model (HLM) for Sentiment Analysis in Code-mixed data.* EMNLP. https://aclanthology.org/2025.emnlp-main.749/
23. Sheth et al. (2026). *Beyond Monolingual Assumptions: A Survey on Code-Switched NLP in the Era of LLMs.* ACL Long. https://aclanthology.org/2026.acl-long.386/ · arXiv:2510.07037
24. Nazir, Bilal et al. (2026). *Sentiment analysis for code-mixed low-resource languages: a systematic literature review.* Social Network Analysis & Mining. https://link.springer.com/article/10.1007/s13278-026-01588-2
25. Bhat et al. (2026). *An exhaustive review of recent advancements in sentiment analysis on Dravidian languages.* Discover Applied Sciences. https://link.springer.com/article/10.1007/s42452-026-08285-x
26. Karunanidhi & Arumugam (2026). *CHMOD_777: LLM-Augmented Transformer Fine-tuning for Tamil Political SA.* https://aclanthology.org/2026.dravidianlangtech-1.23/
27. Nguyen & Văn (2026). *PhucNguyen: Political Multiclass SA with XLM-R + LoRA.* https://aclanthology.org/2026.dravidianlangtech-1.49/
28. Mitharshana & Shanthi (2026). *ByteBuilders: Transformer-Based Weighted Ensemble.* https://aclanthology.org/2026.dravidianlangtech-1.20/
29. Long, Wang & Tsvetkov (2024). *TransliCo: A Contrastive Learning Framework to Address the Script Barrier.* ACL. https://aclanthology.org/2024.acl-long.136/
30. Pednekar & Prakash et al. (2025). *SenTAS++: Language-Tag Graph Attention and Cross-View Contrastive Fusion.* IEEE. https://ieeexplore.ieee.org/abstract/document/11331076
31. *AdapterFusion-based multi-task learning for code-mixed and code-switched text.* Engineering Applications of AI (2023). https://www.sciencedirect.com/science/article/pii/S0952197623014239
32. Xie et al. (2025). *Enhancing Multilingual Language Models for Code-Switched Input Data.* arXiv:2503.07990
33. Singh et al. (2021). *Offence Detection in Dravidian Languages using Code-Mixing-Index-based Focal Loss.* arXiv:2111.06916
34. *FedPerX: A Personalized and Privacy-Preserving Federated Transformer Framework.* IEEE (2025). https://ieeexplore.ieee.org/document/11391653
35. CoLI@FIRE2024 / WILD@FIRE2025 word-level code-mixed LID tasks. https://dl.acm.org/doi/full/10.1145/3734947.3735663
36. FIRE 2022 — *Findings: Sentiment Analysis and Homophobia Detection.* https://dlnext.acm.org/doi/pdf/10.1145/3574318.3574347
37. Chattu et al. (2025). *Sentiment classification for Telugu using transformation-based approaches.* Scientific Reports. https://link.springer.com/article/10.1007/s42001-025-00434-3
38. *Transformer Based Sentiment Analysis on Code Mixed Data.* Procedia CS (2024). https://www.sciencedirect.com/science/article/pii/S1877050924006173
39. AI4Bharat *IndicXlit* / *IndicLID* toolkits (transliteration & language ID).
40. *Findings of the Shared Task on Caste and Migration Hate Speech Detection.* LTEDI (2025). https://aclanthology.org/anthology-files/pdf/ltedi/2025.ltedi-1.34.pdf

*(~70 primary sources were consulted; the 40 above carry the load-bearing claims of this review.)*
