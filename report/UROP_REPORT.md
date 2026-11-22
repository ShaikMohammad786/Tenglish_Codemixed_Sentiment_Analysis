<!-- University logo placeholder -->

# The Capacity Illusion: A Leakage-Audited Benchmark of 21 Models for Romanized Telugu–English Sentiment Analysis

**UROP Project submitted to the**

**SRM University – AP, Andhra Pradesh**

for the partial fulfillment of the requirements to award the degree of

**Bachelor of Technology**

**in**

**Computer Science and Engineering**

**School of Engineering and Sciences**

**Submitted by**

**Shaik Mohammad | AP23110011178**

**Shaik Abdullah Tayyeb | AP23110011186**

**Jaya Krishna | AP23110011205**

**Under the Guidance of**

**Dr. Yandrapati Prakash Babu**

**SRM University–AP**

**Neerukonda, Mangalagiri,**

**Guntur, Andhra Pradesh – 522 240**

**November 2026**

---

# Certificate

This is to certify that the project report entitled **“The Capacity Illusion: A Leakage-Audited Benchmark of 21 Models for Romanized Telugu–English Sentiment Analysis”**, submitted by **Shaik Mohammad (AP23110011178), Shaik Abdullah Tayyeb (AP23110011186), and Jaya Krishna (AP23110011205)** to **SRM University–AP, Andhra Pradesh**, in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering**, is a record of bonafide work carried out by the candidates under my supervision and guidance. The work has not been submitted elsewhere for the award of any other degree or diploma.

**Dr. Yandrapati Prakash Babu**  
Project Guide  
Department of Computer Science and Engineering  
SRM University–AP

**Date:** November 2026

---

# Acknowledgements

I would like to express my sincere gratitude to my mentor, **Dr. Yandrapati Prakash Babu**, for the guidance, patience, and encouragement provided throughout this project. The discussions, careful reviews, and constructive criticism helped shape both the experiments and the final report. I am also grateful to **SRM University–AP** and the **School of Engineering and Sciences** for providing the academic environment and computing resources that made this work possible.

I thank the authors of the **CMTE-NITANP** and **CMTET** datasets for releasing their corpora and making reproducible research on code-mixed Telugu–English sentiment analysis possible. I also acknowledge the open-source community behind PyTorch, Hugging Face Transformers, scikit-learn, and the numerous tools used in this study. Finally, I thank my team members and peers for their constant support, discussions, and help during the preparation of this report.

---

# Table of Contents

- [Certificate](#certificate)
- [Acknowledgements](#acknowledgements)
- [Abstract](#abstract)
- [Abbreviations](#abbreviations)
- [List of Figures](#list-of-figures)
- [List of Tables](#list-of-tables)

1. [Introduction](#1-introduction)
2. [Literature Survey](#2-literature-survey)
3. [Dataset and Leakage Audit](#3-dataset-and-leakage-audit)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Robustness, Calibration, and Transfer](#6-robustness-calibration-and-transfer)
7. [Discussion: Why Do Linear Models Match Transformers?](#7-discussion-why-do-linear-models-match-transformers)
8. [Concluding Remarks](#8-concluding-remarks)
9. [Future Work](#9-future-work)
10. [References](#10-references)
11. [Appendix: Implementation Artifacts](#appendix-implementation-artifacts)

---

# Abstract

Code-mixed Romanized Telugu–English (Tenglish) is a low-resource, orthographically noisy sentiment task. CMTE-NITANP, a 34,588-comment corpus, is a standard benchmark, yet published comparisons mix random and duplicate-aware splits, different tokenizers, and different tuning budgets. This report presents a leakage-audited benchmark of **21 model configurations and 85 final runs** on a group-disjoint split (Split B) of CMTE-NITANP. The matrix spans 11 classical learners, three deep networks, and seven transformer configurations: MuRIL (LoRA and full fine-tuning), plus LoRA for mBERT, Distil-mBERT, XLM-R, IndicBERTv2, and TeluguBERT.

The best single-view model is a calibrated LinearSVC on a word+character union (379 K coefficients): **83.63 macro-F1 and 83.64 accuracy**, fitted end-to-end in **4.6 s**. The best transformer, fully fine-tuned MuRIL (238 M parameters), reaches **83.36 macro-F1 and 83.38 accuracy** after **53.9 min**. The 0.27-point difference is not statistically significant (McNemar p = 0.659; 95% CI [−0.73, +1.21] percentage points). Parameter count does not predict accuracy (Spearman ρ = 0.306, p = 0.504). A post-hoc multi-view linear variant that adds PLD-normalized features reaches **83.92/83.93**; a six-model heterogeneous ensemble reaches **85.00/85.01** (temperature-calibrated 85.02/85.03), but the ensemble is selection-sensitive: strict validation-only selection gives **84.77**, and a validation-fitted simplex blend gives **84.33**.

A random split contains 345 exact Jaccard ≥ 0.8 train–test pairs and 203 test rows with a near-duplicate in train; the group-disjoint split contains zero. Feature ablation shows the signal is predominantly unigram-lexical: word bigrams do not reliably help the linear models, and WordPiece tokenization does not close the gap. Under orthographic perturbation, vowel elongation and combined noise cause the largest drops; zero-shot transfer to CMTET loses 17.54–25.70 macro-F1 points. We argue that the transformer advantage on this task is largely a **capacity illusion**: once leakage and tuning are controlled, a lexical linear model is statistically indistinguishable from a 238 M-parameter model at a small fraction of the cost.

**Keywords:** code-mixed sentiment analysis, Telugu–English, benchmark, data leakage, linear models, transformers, parameter-efficient fine-tuning.

---

# Abbreviations

| Abbreviation | Meaning |
|---|---|
| CMTE-NITANP | Code-Mixed Telugu–English corpus, NIT Andhra Pradesh |
| CMTET / CMTE-IIITH | Code-Mixed Telugu–English Text corpus, IIIT Hyderabad |
| CMSA | Code-Mixed Sentiment Analysis |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| LoRA | Low-Rank Adaptation |
| PEFT | Parameter-Efficient Fine-Tuning |
| MLF | Matrix Language Frame |
| ECE | Expected Calibration Error |
| MCC | Matthews Correlation Coefficient |
| OOV | Out-of-Vocabulary |
| CI | Confidence Interval |
| PLD | Phonetic Levenshtein Distance |
| GPU | Graphics Processing Unit |
| UROP | Undergraduate Research Opportunities Programme |

---

# List of Figures

| Figure | Caption | File |
|---|---|---|
| Figure 1 | Leakage audit: Split A vs Split B | `figures/fig_split_audit.png` |
| Figure 2 | Macro-F1 by model family | `figures/01_model_comparison.png` |
| Figure 3 | Parameter count vs macro-F1 | `figures/fig_capacity.png` |
| Figure 4 | Per-class F1 | `figures/02_per_class_f1.png` |
| Figure 5 | Accuracy versus training time | `figures/06_cost_benefit.png` |
| Figure 6 | Orthographic robustness | `figures/fig_robustness.png` |
| Figure 7 | Reliability diagrams | `figures/04_reliability.png` |
| Figure 8 | Zero-shot transfer to CMTET | `figures/fig_cross_corpus.png` |
| Figure 9 | Hyperparameter sensitivity | `figures/08_sensitivity.png` |
| Figure 10 | Risk–coverage curves | `figures/05_risk_coverage.png` |

# List of Tables

| Table | Caption |
|---|---|
| Table 1 | CMTE-NITANP Split B statistics |
| Table 2 | Model matrix and frozen hyperparameters |
| Table 3 | Final leaderboard on Split B |
| Table 4 | Paired comparisons against logistic regression |
| Table 5 | Feature ablation |
| Table 6 | Orthographic robustness |
| Table 7 | Zero-shot transfer to CMTET |
| Table 8 | Transformer hyperparameter sensitivity |
| Table 9 | Truncation at different maximum lengths |
---

# 1. Introduction

Code-mixing is the alternation of two or more languages within a single utterance. In Telugu–English code-mixing, often written in Roman script (*Tenglish*), speakers mix English words with Romanized Telugu morphology. Examples include “movie chala bagundi” (“the movie is very good”) and “trailer keka bro” (“the trailer is superb, brother”). Sentiment analysis of such text is difficult because the text is short, spelling is highly variable, and the same sentiment may be expressed through Telugu, English, or a hybrid construction.

CMTE-NITANP is a 34,588-comment Telugu–English sentiment corpus with three classes: negative (12,583), neutral (10,944), and positive (11,061). Recent work reports that transformer models improve over classical baselines by about 5% accuracy and 6% F1 on this corpus. That claim is plausible, but it is difficult to evaluate: the corpus contains substantial near-duplication, and reported experiments differ in split construction, tokenizer, and tuning budget. On a small, noisy, lexically driven task, each of these choices can move macro-F1 by an amount comparable to the reported transformer advantage.

This report asks a deliberately narrow question: **on a leak-free split, with tuning held fixed, does a 238 M-parameter transformer actually beat a linear model on CMTE-NITANP?** We answer it with a controlled benchmark of 21 configurations and 85 final runs, spanning classical learners, deep networks, LoRA adapters, and full fine-tuning. We also audit the split itself with exact Jaccard matching, because near-duplicate leakage is one of the most common explanations for inflated social-media results.

## 1.1 Research Questions

- **RQ1 (ranking):** Which model family attains the best macro-F1 on a leak-free split?
- **RQ2 (linear vs transformer):** Is the best transformer significantly better than the best linear model?
- **RQ3 (capacity):** Does parameter count predict macro-F1 across the matrix?
- **RQ4 (robustness):** How much does orthographic noise degrade each model?
- **RQ5 (cost):** What macro-F1 is obtained per second of training?
- **RQ6 (calibration):** Which models produce reliable confidence estimates?
- **RQ7 (transfer):** How well do models transfer zero-shot to CMTET?
- **RQ8 (abstention):** Does confidence-based abstention buy a useful risk–coverage trade-off?

## 1.2 Contributions

1. **A leakage-audited split.** We cluster the corpus with MinHash/LSH over word 3-grams, build a group-disjoint Split B, and verify it with an exact Jaccard pass. Split A contains 345 train–test pairs with exact Jaccard ≥ 0.8 and 203 test rows with a near-duplicate in train; Split B contains zero.
2. **A 21-configuration, 85-run benchmark.** We compare 11 classical learners, three deep networks, six LoRA adapters, and MuRIL full fine-tuning under a search-once–freeze–then-seeds protocol. Seeds change only initialization and shuffling, never learning rate, batch size, or sequence length.
3. **A mechanistic decomposition.** We isolate the effects of feature type (word unigram, word bigram, character n-gram, union), tokenization (word vs WordPiece), adaptation regime (LoRA vs full fine-tuning), calibration, orthographic robustness, cross-corpus transfer, and hyperparameter sensitivity.

## 1.3 Report Organisation

Section 2 reviews related work. Section 3 describes the corpus, preprocessing, and leakage audit. Section 4 presents the experimental protocol. Section 5 reports the main results and statistical tests. Section 6 covers robustness, calibration, transfer, and sensitivity. Section 7 explains why the linear model matches the transformer. Sections 8–9 give concluding remarks and future work.

---

# 2. Literature Survey

## 2.1 Code-Mixed Sentiment Analysis

Sentiment analysis of code-mixed Indian social media was popularised by shared tasks such as SemEval-2020 Task 9 [5] and the DravidianLangTech / DravidianCodeMix series [4]. Telugu–English resources include CMTE-IIITH [3] and CMTE-NITANP [1]. Rayala and Seshadri introduced syllable-aware word embeddings for Telugu–English [1] and later evaluated transformer models on both corpora [2]. Their abstract reports a 5% accuracy and 6% F1 improvement over baselines on CMTE-NITANP.

## 2.2 Transformers and Parameter-Efficient Fine-Tuning

Multilingual encoders—mBERT [6], XLM-R [7], MuRIL [8], IndicBERT [9], and TeluguBERT—are the default starting point for low-resource Indic tasks. Full fine-tuning updates all parameters; LoRA [10] freezes the base model and learns low-rank updates to selected projections, and 8-bit optimisers [11] make full fine-tuning feasible on small GPUs. Parameter-efficient methods are attractive when compute is scarce, but their capacity is bounded by the rank and target modules. Whether that bound matters on a short, lexical task is an empirical question, and it is the question our LoRA-versus-full-fine-tuning comparison isolates.

## 2.3 Linear and Lexical Baselines

TF-IDF with linear classifiers remains a strong baseline for short-text sentiment [20], [21]. Character n-grams are robust to spelling variation [22], which is pervasive in Romanized Telugu. Linear models have high bias and low variance; on a small corpus with strong lexical cues, this bias can be an advantage rather than a limitation. Ensemble and boosting methods (random forests [14], gradient boosting [15], XGBoost [16], LightGBM [17]) are common alternatives, but they add capacity without adding lexical knowledge.

## 2.4 Leakage and Evaluation Rigour

Near-duplicate leakage is a well-known failure mode in data mining [23]: random splits place near-identical rows in both training and test, inflating estimates. Social-media corpora are especially vulnerable because retweets, copy-paste comments, and spelling variants create large near-duplicate clusters. Group-aware splitting removes this path. For significance testing on a single fixed test set, paired McNemar tests [24], Holm correction [25], and paired bootstrap confidence intervals [26] are appropriate; the Nadeau–Bengio correction [27] is designed for resampled cross-validation and is not used here. Calibration is measured with expected calibration error (ECE) and the Brier score [28].

## 2.5 Research Gap

Prior work compares transformers with untuned or weakly tuned classical baselines on random splits. It rarely reports a leakage audit, a tuned linear baseline with character features, a parameter-efficient adaptation comparison, or a robustness/calibration analysis. This report fills that gap with a controlled, reproducible matrix.

---

# 3. Dataset and Leakage Audit

## 3.1 Corpus

CMTE-NITANP contains 34,588 Romanized Telugu–English comments with three labels: negative (12,583), neutral (10,944), and positive (11,061). Comments are short: mean 10.46 whitespace tokens (median 8, 95th percentile 30, maximum 148) and mean 66.7 characters (median 47). The training vocabulary contains 45,868 word types; 11.12% of test tokens are unseen in training. These statistics already suggest that the task is local and lexical rather than long-range.

**Table 1. CMTE-NITANP Split B statistics.**

| Split | Rows | Negative | Neutral | Positive | Tokens (mean / median) | Characters (mean / median) |
|---|---:|---:|---:|---:|---|---|
| Train | 27,670 | 10,066 | 8,755 | 8,849 | 10.46 / 8 | 66.7 / 47 |
| Validation | 3,459 | 1,259 | 1,094 | 1,106 | 10.55 / 7 | 67.3 / 47 |
| Test | 3,459 | 1,258 | 1,095 | 1,106 | 10.50 / 7 | 66.9 / 47 |
| **Total** | **34,588** | **12,583** | **10,944** | **11,061** | — | — |

*Token and character counts are computed on `text_clean`, the field used by every main-table experiment.*

## 3.2 Preprocessing

We apply a deterministic, signal-preserving pipeline. Unicode is normalised to NFC; HTML entities are unescaped; URLs and @mentions are replaced by placeholders; the `#` of a hashtag is dropped while the word is kept; whitespace is collapsed; and text is lowercased. Emojis, punctuation, digits, and character elongations are **not** removed in the main matrix, because they carry sentiment and because removing them would confound the robustness analysis.

The corpus also ships with a 9,388-entry PLD variant→canonical dictionary; applying it word by word changes 20,414 of 27,670 training rows (73.8%), which confirms the orthographic variability of the source. The main experiments use `text_clean` (before PLD normalisation) so that all models see the same input; the normalised field is retained for the post-hoc multi-view experiment in Section 5.1.

## 3.3 Near-Duplicate Clustering and Splits

We cluster the 34,588 rows with MinHash/LSH over word 3-grams (128 permutations, Jaccard threshold 0.8). This yields 33,313 clusters; 2,283 rows (6.60%) belong to a cluster of size > 1. Among multi-row clusters, 115 (11.41%) contain conflicting labels, covering 314 rows, with mean label entropy 0.113 bits—a small but non-zero annotation-noise floor.

We construct two splits of identical size (27,670 / 3,459 / 3,459):

- **Split A (control, leaky):** stratified random 80/10/10, ignoring cluster membership. 203 test rows (5.87%) have a near-duplicate in training.
- **Split B (treatment, leak-free):** StratifiedGroupKFold with 10 folds, so an entire cluster lands in one split; whole clusters are then moved between train/validation/test to match Split A’s sizes. 0 test rows have a near-duplicate in training.

Split B is a genuine regroup, not Split A with twins moved: 19.99% of training rows, 90.32% of validation rows, and 89.65% of test rows change membership relative to Split A. An independent exact-Jaccard audit confirms the MinHash result: Split A has 345 train–test pairs with exact Jaccard ≥ 0.8 (maximum 1.0), while Split B has 0. The audit is stable across thresholds 0.7, 0.8, and 0.9 (Split A leakage 5.81–6.10%, Split B 0.00%). All experiments below use Split B.

![Figure 1: Leakage audit. Left: exact Jaccard ≥ 0.8 train–test pairs. Right: test rows with a near-duplicate in train. Split B removes both.](figures/fig_split_audit.png)

## 3.4 Why the Audit Matters

A random split can make a memorising model look better than it is. The 345 exact pairs in Split A are not a theoretical concern: comments such as “trailer keka bro” and “jai balayya” repeat across train and test. A model that memorises these phrases gains an advantage that will not transfer to new comments. Split B removes this path, and all reported numbers use it.

---

# 4. Methodology

## 4.1 Model Matrix

The matrix contains 21 configurations in three families:

- **Classical (11):** logistic regression, linear SVM [18], multinomial Naive Bayes, k-NN [19], dummy majority, random forest, extra trees, histogram gradient boosting, XGBoost, LightGBM, and a soft-voting ensemble of logistic regression, random forest, extra trees, and histogram gradient boosting.
- **Deep (3):** TextCNN, BiLSTM, and CNN+BiLSTM, trained from scratch on a word-level vocabulary built from the training split (minimum frequency 2), with 100-dimensional embeddings, dropout 0.3, Adam, learning rate 10⁻³, batch 128, maximum length 64, and the same 10-epoch/patience-2 schedule.
- **Transformers (7):** MuRIL-base (LoRA and full fine-tuning), mBERT (LoRA), Distil-mBERT (LoRA), XLM-R-base (LoRA), IndicBERTv2 (LoRA), and TeluguBERT (LoRA).

Classical and deep models are implemented with scikit-learn [13] and PyTorch; transformer models use Hugging Face Transformers [12] and PEFT.

**Table 2. Model matrix and frozen hyperparameters.**

| Family | Configurations | Seeds | Key hyperparameters |
|---|---|---:|---|
| Classical | LogReg, LinearSVC, MNB, k-NN, Dummy, RF, ET, HGB, XGB, LGBM, Soft voting | 5 (k-NN 1) | TF-IDF `min_df=2`, sublinear TF; word unigrams; LogReg C=1.0; LinearSVC C=0.1 with Platt calibration; 300 trees; boosting grids lr {0.05, 0.1}, 200–400 iterations |
| Deep | TextCNN, BiLSTM, CNN+BiLSTM | 5 | 100-d embeddings, dropout 0.3, Adam 10⁻³, batch 128, max_len 64, 10 epochs, patience 2 |
| Transformers | MuRIL LoRA, MuRIL full-FT, mBERT LoRA, Distil-mBERT LoRA, XLM-R LoRA, IndicBERTv2 LoRA, TeluguBERT LoRA | 3 (full-FT 1) | LoRA r=16, α=32, q/v, lr 4×10⁻⁴, max_len 64, batch 16×2, 10 epochs, patience 2; full-FT lr 2×10⁻⁵, 8-bit AdamW, batch 8×4, gradient checkpointing |

## 4.2 Search-Once, Freeze, Then Vary Seeds

Every hyperparameter is selected once on the validation split with seed 42, frozen to a JSON file, and then reused unchanged for all final seeds. Seeds vary only weight initialisation, data order, and dropout masks; they never change learning rate, batch size, sequence length, LoRA rank, or epoch budget. This protocol directly addresses a common benchmark flaw in which “seed runs” silently re-tune the model. Classical models use five seeds (k-NN uses one; deterministic learners have zero variance across seeds); deep models use five seeds; LoRA transformers use three seeds (42–44); MuRIL full fine-tuning uses one seed because of its 53.9-minute cost on the available GPU.

## 4.3 Hyperparameters

TF-IDF uses `min_df=2` and sublinear term frequency. The primary feature is word unigrams (16,892 features); the ablation adds word bigrams (40,604), character n-grams (2–5, 85,764), and their union (126,368). Logistic regression uses C=1.0 and the linear SVM uses C=0.1 (each selected on validation). Tree ensembles use 300 trees; boosting grids cover learning rate {0.05, 0.1} and 200–400 iterations.

LoRA uses r=16, α=32, dropout 0.1 on the query and value projections, learning rate 4×10⁻⁴, weight decay 0.01, warmup 0.06, batch 16 with gradient accumulation 2 (effective 32), maximum length 64, 10 epochs, patience 2, minimum 3 epochs, and best-on-validation restoration. MuRIL full fine-tuning uses learning rate 2×10⁻⁵, 8-bit AdamW, batch 8 with accumulation 4 (effective 32), gradient checkpointing, and the same schedule. Linear SVM probabilities are obtained by Platt scaling with 3-fold cross-validation on the training split (`CalibratedClassifierCV`), so its ECE and Brier scores are comparable to the probabilistic models. The full-fine-tuning run uses 237,558,531 trainable parameters; LoRA adapters use 592,131 (MuRIL, mBERT, IndicBERT, TeluguBERT), 887,811 (Distil-mBERT), and 1,182,723 (XLM-R).

## 4.4 Metrics and Statistics

The primary metric is macro-F1 on the 3,459-row test set. We also report accuracy, weighted-F1, MCC, ECE, Brier score, and risk–coverage curves. Confidence intervals are 95% paired bootstrap intervals (1,000 resamples, seed 42) over the test set. Model comparisons use the continuity-corrected McNemar chi-square test on the paired correctness vectors (exact binomial when discordant pairs are fewer than 25), with Holm–Bonferroni correction across the 20 comparisons against logistic regression. ECE uses 10 equal-width bins. McNemar tests accuracy, whereas the bootstrap interval is for macro-F1; the two can disagree marginally (for example, Linear SVM’s macro-F1 interval excludes zero while its McNemar p is 0.062). We therefore treat the Holm-corrected McNemar test as the primary significance test and report the macro-F1 interval as descriptive. We do not use the Nadeau–Bengio correction because there is a single fixed split, not repeated cross-validation. All 85 final runs are recomputed from stored prediction files; the recomputation produced zero mismatches.

## 4.5 Compute

All runs use one NVIDIA RTX 2050 laptop GPU with 4 GB of memory (3.87 GB usable) and a parallel CPU lane. The longest run is MuRIL full fine-tuning at 3,231.7 s (53.9 min); logistic regression takes 10.4 s, linear SVM 3.6 s, and multinomial Naive Bayes 0.2 s. The best single linear union takes 4.6 s end-to-end; full fine-tuning therefore costs roughly 700× more compute for a nominally lower score.

---

# 5. Results

## 5.1 Main Leaderboard (RQ1)

**Table 3. Final leaderboard on CMTE-NITANP Split B.** Values are mean ± standard deviation over final seeds; macro-F1 and accuracy are percentages. The last two rows are post-hoc linear variants outside the frozen 21-configuration matrix.

| # | Model | Family | Seeds | Macro-F1 | 95% CI | Accuracy | ECE | Parameters | Train (s) |
|---:|---|---|---:|---:|---|---:|---:|---|---:|
| 1 | MuRIL full-FT | Transformer | 1 | 83.36 ± 0.00 | [82.11, 84.67] | 83.38 | 0.0702 | 237.6 M | 3231.7 |
| 2 | Logistic regression | Classical | 5 | 82.93 ± 0.00 | [81.82, 84.19] | 82.91 | 0.0680 | 50.7 K | 10.4 |
| 3 | Linear SVM | Classical | 5 | 82.31 ± 0.00 | [81.17, 83.66] | 82.31 | 0.0431 | 50.7 K | 3.6 |
| 4 | Soft voting | Classical | 5 | 82.31 ± 0.07 | [81.12, 83.48] | 82.29 | 0.0677 | — | 242.2 |
| 5 | TeluguBERT LoRA | Transformer | 3 | 81.96 ± 0.43 | [80.26, 82.87] | 81.93 | 0.0634 | 238.2 M (0.59 M) | 1562.3 |
| 6 | MuRIL LoRA | Transformer | 3 | 81.65 ± 0.80 | [81.14, 83.54] | 81.67 | 0.0540 | 238.2 M (0.59 M) | 1246.9 |
| 7 | IndicBERT LoRA | Transformer | 3 | 81.31 ± 0.25 | [80.10, 82.64] | 81.31 | 0.0589 | 278.6 M (0.59 M) | 1881.6 |
| 8 | Multinomial NB | Classical | 5 | 81.21 ± 0.00 | [79.92, 82.53] | 81.30 | 0.0861 | 50.7 K | 0.2 |
| 9 | XLM-R LoRA | Transformer | 3 | 81.03 ± 0.66 | [79.00, 81.67] | 81.02 | 0.0547 | 279.2 M (1.18 M) | 1365.2 |
| 10 | mBERT LoRA | Transformer | 3 | 80.98 ± 0.44 | [79.35, 82.08] | 80.99 | 0.0657 | 178.4 M (0.59 M) | 1391.0 |
| 11 | Distil-mBERT LoRA | Transformer | 3 | 80.92 ± 0.22 | [79.46, 81.96] | 80.93 | 0.0541 | 136.2 M (0.89 M) | 657.8 |
| 12 | Extra trees | Classical | 5 | 80.74 ± 0.17 | [79.52, 82.17] | 80.72 | 0.0548 | — | 63.8 |
| 13 | LightGBM | Classical | 5 | 80.48 ± 0.00 | [79.28, 81.74] | 80.49 | 0.0132 | — | 338.1 |
| 14 | Hist. gradient boost. | Classical | 5 | 80.18 ± 0.13 | [78.97, 81.57] | 80.16 | 0.0220 | — | 64.7 |
| 15 | BiLSTM | Deep | 5 | 79.83 ± 0.42 | [78.97, 81.63] | 79.83 | 0.0603 | — | 9.6 |
| 16 | Random forest | Classical | 5 | 79.68 ± 0.22 | [78.26, 80.84] | 79.70 | 0.0429 | — | 39.0 |
| 17 | TextCNN | Deep | 5 | 79.28 ± 0.36 | [77.57, 80.23] | 79.29 | 0.0872 | — | 15.5 |
| 18 | CNN+BiLSTM | Deep | 5 | 78.46 ± 0.40 | [77.00, 79.60] | 78.46 | 0.0641 | — | 40.0 |
| 19 | XGBoost | Classical | 5 | 77.70 ± 0.24 | [76.40, 79.10] | 77.71 | 0.0759 | — | 41.2 |
| 20 | k-NN | Classical | 1 | 77.55 ± 0.00 | [76.22, 78.97] | 77.59 | 0.0665 | — | 2.2 |
| 21 | Dummy majority | Classical | 5 | 17.78 ± 0.00 | [17.22, 18.35] | 36.37 | 0.6363 | — | 0.1 |
| — | **LinearSVC union (single-view, post-hoc)** | Classical | 1 | **83.63** | — | **83.64** | 0.0372 | 379.1 K | 4.6 |
| — | **LinearSVC clean+PLD (multi-view, post-hoc)** | Classical | 1 | **83.92** | [82.74, 85.17] | **83.93** | 0.0414 | 752.4 K | 12.2 |

Within the frozen matrix, MuRIL full-FT is nominally first at 83.36/83.38. Outside the matrix, the calibrated LinearSVC union is the best single-view model at **83.63/83.64**, nominally ahead of full-FT by 0.27 points; the difference is not significant (McNemar p = 0.659; 95% CI [−0.73, +1.21] percentage points). The post-hoc multi-view variant reaches **83.92/83.93** (p = 0.275 vs the single-view union; p = 0.307 vs full-FT), but it is not robustness-tested.

![Figure 2: Macro-F1 by model family on the leak-free Split B. Full fine-tuned MuRIL is nominally first in the frozen matrix; the post-hoc linear union is nominally ahead.](figures/01_model_comparison.png)

## 5.2 Linear vs Transformer (RQ2)

The paired comparison is consistent across tests. Full fine-tuning versus the deterministic logistic regression gives a test macro-F1 difference of +0.43 points (paired-bootstrap Δ = +0.39; 95% CI [−0.62, +1.40] percentage points; McNemar p = 0.437; Holm p = 0.816). The direct best-single comparison is even tighter: the calibrated LinearSVC union is nominally ahead of full-FT by **+0.27 points** (+0.26 accuracy), with McNemar p = 0.659 and 95% CI [−0.73, +1.21]. Because full fine-tuning has one seed, these intervals are test-set bootstrap intervals, not cross-seed intervals. In contrast, 14 of the 20 comparisons against logistic regression are significant after Holm correction; the exceptions are full fine-tuning, MuRIL LoRA, soft voting, linear SVM, IndicBERT LoRA, and TeluguBERT LoRA (16 are significant at the uncorrected p < 0.05 level). The result is not that transformers are weak; it is that the best linear model and the best transformer are statistically tied, with the linear model nominally ahead.

**Table 4. Paired comparisons against logistic regression (seed 42).** Δ is the paired-bootstrap mean difference in macro-F1 (percentage points); negative means worse than LogReg. Bold rows are not significant after Holm-corrected McNemar (p > 0.05).

| Model | Δ macro-F1 | 95% CI | McNemar p | Holm p |
|---|---:|---|---:|---:|
| **MuRIL full-FT** | +0.39 | [−0.62, +1.40] | 0.4367 | 0.8158 |
| **MuRIL LoRA** | −0.58 | [−1.70, +0.47] | 0.4079 | 0.8158 |
| **Linear SVM** | −0.61 | [−1.20, −0.05] | 0.0622 | 0.2488 |
| **Soft voting** | −0.69 | [−1.53, +0.13] | 0.1495 | 0.4485 |
| **TeluguBERT LoRA** | −1.47 | [−2.63, −0.34] | 0.0203 | 0.1015 |
| **IndicBERT LoRA** | −1.64 | [−2.92, −0.42] | 0.0141 | 0.0846 |
| Multinomial NB | −1.74 | [−2.65, −0.74] | 0.0023 | 0.0161 |
| Distil-mBERT LoRA | −2.21 | [−3.32, −1.04] | 0.0006 | 0.0048 |
| Extra trees | −2.23 | [−3.18, −1.28] | 0.0000 | 0.0000 |
| mBERT LoRA | −2.35 | [−3.56, −1.02] | 0.0003 | 0.0027 |
| LightGBM | −2.49 | [−3.41, −1.56] | 0.0000 | 0.0000 |
| XLM-R LoRA | −2.64 | [−3.89, −1.48] | 0.0000 | 0.0000 |
| BiLSTM | −2.73 | [−3.86, −1.70] | 0.0000 | 0.0000 |
| Hist. gradient boost. | −2.74 | [−3.83, −1.86] | 0.0000 | 0.0000 |
| Random forest | −3.44 | [−4.41, −2.38] | 0.0000 | 0.0000 |
| TextCNN | −4.02 | [−5.11, −2.89] | 0.0000 | 0.0000 |
| CNN+BiLSTM | −4.66 | [−5.76, −3.56] | 0.0000 | 0.0000 |
| XGBoost | −5.23 | [−6.40, −4.16] | 0.0000 | 0.0000 |
| k-NN | −5.41 | [−6.52, −4.24] | 0.0000 | 0.0000 |
| Dummy majority | −65.18 | [−66.45, −63.86] | 0.0000 | 0.0000 |

### 5.2.1 Feature Ablation

**Table 5. Feature ablation: test macro-F1 (%) on Split B, seed 42.** The union is the 126,368-dimensional combined feature space used by the best linear model.

| Model | Word unigram (16,892) | Word bigram (40,604) | Char 2–5 (85,764) | Union (126,368) |
|---|---:|---:|---:|---:|
| Logistic regression | 82.93 | 80.79 | 82.76 | 83.24 |
| Linear SVM | 82.31 | 82.23 | 83.13 | **83.63** |
| Soft voting | 82.27 | 82.14 | 82.31 | 83.18 |
| Multinomial NB | 81.21 | **82.07** | 79.53 | 81.69 |
| LightGBM | 80.48 | 80.52 | 82.34 | 82.77 |
| XGBoost | 77.73 | 77.42 | 80.62 | 80.91 |

Word bigrams are not a reliable gain: they hurt logistic regression by 2.14 points and improve multinomial NB by 0.86; the remaining changes are at most 0.31 points. Character n-grams are more useful for the margin-based and boosted models (Linear SVM +0.82, LightGBM +1.86, XGBoost +2.89 over word unigrams), but they hurt multinomial NB (−1.68). The union is best for five of the six models; multinomial NB peaks on word bigrams (82.07), consistent with its independence assumption. The calibrated LinearSVC union is the best single-view model at 83.63. The pattern supports the interpretation in Section 7: spelling variation is better captured by substrings than by word identity, and a linear model over the union has enough capacity for the cue set.

## 5.3 Capacity Does Not Predict Accuracy (RQ3)

Across the seven transformer configurations, parameter count and macro-F1 are weakly and non-significantly correlated (Spearman ρ = 0.306, p = 0.504). The 136 M-parameter Distil-mBERT (80.92) is within 0.73 points of the 238 M-parameter MuRIL LoRA (81.65), and the 279 M-parameter XLM-R LoRA (81.03) is below both. Meanwhile a 50.7 K-coefficient LogReg (82.93) and a 379 K-coefficient calibrated linear union (83.63) beat every LoRA adapter. Capacity is not the binding constraint; the binding constraint is whether the model can represent the lexical cues, and a linear model can.

![Figure 3: Parameter count versus test macro-F1 for the seven transformer configurations. The dashed line is logistic regression (82.93). The correlation is weak and non-significant (ρ = 0.306, p = 0.504).](figures/fig_capacity.png)

## 5.4 Per-Class Behaviour

Full fine-tuning and logistic regression have almost identical per-class profiles: negative 83.25 vs 82.34, neutral 81.28 vs 80.93, positive 85.55 vs 85.52. Neutral is the hardest class for both, consistent with the absence of a strong lexical polarity cue. The full-fine-tuning difference is concentrated in positive recall (86.17 vs 84.90) and neutral F1 (81.28 vs 80.93); on negative recall it is slightly lower (83.78 vs 84.34). The two models make different errors but reach the same aggregate. The near-identical positive F1 (85.55 vs 85.52) is particularly striking: the 238 M-parameter model buys no measurable improvement on the easiest class.

![Figure 4: Per-class F1 on Split B.](figures/02_per_class_f1.png)

## 5.5 Cost (RQ5)

The compute gap is extreme. The best single linear union reaches 83.63 in **4.6 s** end-to-end; full fine-tuning reaches 83.36 in **3,231.7 s**, roughly **700×** the compute for a nominally **lower** score. Logistic regression reaches 82.93 in 10.4 s. Multinomial Naive Bayes reaches 81.21 in 0.2 s, the best F1-per-second in the matrix. The cost–benefit frontier is populated by classical models up to 10.4 s; the only transformer near the frontier is full fine-tuning at 3,231.7 s, which is 0.27 points below the best linear union.

![Figure 5: Accuracy versus training time. Classical models populate the frontier up to 10.4 s; full fine-tuning is the only transformer at the far right.](figures/06_cost_benefit.png)

## 5.6 RQ Summary

| RQ | Answer |
|---|---|
| RQ1 | Best frozen-matrix model: MuRIL full-FT 83.36; best overall single-view: calibrated LinearSVC union 83.63 |
| RQ2 | Best linear vs best transformer: +0.27 points, McNemar p = 0.659; not significant |
| RQ3 | Parameter count does not predict macro-F1 (ρ = 0.306, p = 0.504) |
| RQ4 | Vowel elongation and combined noise dominate; linear union is most robust (combined Δ = 10.19 points) |
| RQ5 | Linear union 4.6 s vs full-FT 3,231.7 s; ~700× compute for no significant gain |
| RQ6 | LightGBM best-calibrated (ECE 0.0132); full-FT ECE 0.0702 |
| RQ7 | IndicBERT best zero-shot CMTET (F1 63.80; drop 17.54 points) |
| RQ8 | Abstention raises macro-F1 monotonically; full-FT and LogReg curves are close |

---

# 6. Robustness, Calibration, and Transfer

## 6.1 Orthographic Robustness (RQ4)

We apply four perturbations to the test set: vowel elongation (e.g., *bagundiii*), voicing confusion (g↔v), aspiration removal (kh→k, th→t, etc.), and their combination. Aspiration is nearly harmless (Δ ≤ 0.23 points). Voicing costs 1.0–2.1 points. Vowel elongation is severe: **8.24–20.66 points** across models (8.24 for the calibrated linear union; 13.21–20.66 for the transformers). The combined perturbation is worst: **10.19–23.61 points** (10.19 for the linear union; 15.99–23.61 for the transformers). Among the transformers, Distil-mBERT has the smallest worst-case degradation (15.99) and IndicBERT the largest (23.61); the calibrated linear union is the most robust overall (combined Δ = 10.19), even though it is the best single model. Full fine-tuning is not more robust than LoRA (17.65 vs 18.94 for MuRIL LoRA), so robustness is not a simple function of adaptation capacity. The practical implication is that vowel normalisation should be part of the preprocessing pipeline, not an afterthought.

**Table 6. Orthographic robustness: test macro-F1 (%) on clean text and four perturbed versions.** Δ is the drop from clean, in percentage points. The clean column is the seed-42 checkpoint used for the perturbation run; it can differ from the seed-averaged leaderboard in Table 3 (for example, MuRIL LoRA 82.42 here vs 81.65 there).

| Model | Clean | Vowel F1 | Δ | Voicing F1 | Δ | Aspiration F1 | Δ | Combined F1 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MuRIL full-FT | 83.36 | 68.22 | 15.14 | 81.70 | 1.66 | 83.22 | 0.14 | 65.71 | 17.65 |
| MuRIL LoRA | 82.42 | 67.06 | 15.36 | 80.50 | 1.92 | 82.28 | 0.14 | 63.48 | 18.94 |
| TeluguBERT LoRA | 81.51 | 63.90 | 17.61 | 80.51 | 1.00 | 81.39 | 0.12 | 61.22 | 20.30 |
| IndicBERT LoRA | 81.34 | 60.68 | 20.66 | 79.88 | 1.46 | 81.16 | 0.17 | 57.73 | 23.61 |
| mBERT LoRA | 80.69 | 66.67 | 14.02 | 78.70 | 1.99 | 80.47 | 0.22 | 64.54 | 16.15 |
| Distil-mBERT | 80.66 | 66.60 | 14.06 | 78.53 | 2.13 | 80.50 | 0.16 | 64.67 | 15.99 |
| XLM-R LoRA | 80.37 | 67.16 | 13.21 | 79.07 | 1.31 | 80.15 | 0.23 | 64.02 | 16.36 |
| **LinearSVC union** | **83.63** | **75.39** | **8.24** | **82.51** | **1.12** | **83.55** | **0.08** | **73.44** | **10.19** |

![Figure 6: Orthographic robustness of the seven transformers. The calibrated linear union (Table 6, last row) is the most robust overall.](figures/fig_robustness.png)

## 6.2 Calibration (RQ6)

LightGBM is the best-calibrated model (ECE 0.0132), followed by histogram gradient boosting (0.0220) and random forest (0.0429). The transformers are poorly calibrated (MuRIL full-FT 0.0702, mBERT 0.0657, TeluguBERT 0.0634, MuRIL LoRA 0.0540), and logistic regression is 0.0680. The pattern is consistent with the literature in part: the boosted trees are best calibrated, while the transformers and logistic regression are less well calibrated; calibration is not simply a function of capacity. For a deployment that thresholds probabilities or abstains, this matters more than a 0.4-point macro-F1 difference.

![Figure 7: Reliability diagrams on Split B. LightGBM is the best-calibrated model (ECE 0.0132); the transformers are systematically over-confident.](figures/04_reliability.png)

## 6.3 Zero-Shot Transfer (RQ7)

Zero-shot transfer from CMTE-NITANP to CMTET (19,868 rows, different annotation scheme and domain) is shown in Table 7. IndicBERT is the best transfer model (63.80), followed by TeluguBERT (63.34) and MuRIL full-FT (63.30). The drop from in-domain performance is 17.54–25.70 points; Distil-mBERT transfers worst (55.03, drop 25.70). Two patterns are notable. First, in-domain rank does not predict transfer rank: MuRIL full-FT is the best in-domain model but third in transfer, while IndicBERT is fourth in-domain but first in transfer. Second, all models degrade sharply on the conflicting-label subset of CMTET (34.07–37.57), which indicates that label-scheme and annotation differences, not just domain shift, are responsible. Transfer is therefore not a free proxy for robustness.

**Table 7. Zero-shot transfer from CMTE-NITANP (Split B) to CMTET.** Drop is in-domain macro-F1 minus CMTET macro-F1, in percentage points. In-domain values are the seed-42 checkpoints used for the transfer run and can differ from the seed-averaged leaderboard.

| Model | In-domain | CMTET | Drop | CMTET Acc. | Conflict |
|---|---:|---:|---:|---:|---:|
| IndicBERT LoRA | 81.34 | **63.80** | 17.54 | 65.51 | 34.72 |
| TeluguBERT LoRA | 81.49 | 63.34 | 18.15 | 64.64 | 37.35 |
| MuRIL full-FT | 83.36 | 63.30 | 20.06 | 64.64 | 36.08 |
| MuRIL LoRA | 82.39 | 62.74 | 19.65 | 64.18 | 36.93 |
| XLM-R LoRA | 80.35 | 59.82 | 20.53 | 61.44 | 34.07 |
| mBERT LoRA | 80.64 | 58.45 | 22.19 | 60.20 | 36.22 |
| Distil-mBERT | 80.73 | 55.03 | 25.70 | 55.50 | 37.57 |

![Figure 8: Zero-shot transfer to CMTET. In-domain rank does not predict transfer rank; IndicBERT transfers best despite being fourth in-domain.](figures/fig_cross_corpus.png)

## 6.4 Hyperparameter Sensitivity

Table 8 reports an OFAT sensitivity study on validation, with a fixed five-epoch budget and seed 42, for MuRIL, mBERT, and Distil-mBERT. Lowering the learning rate to 2×10⁻⁴ produces the weakest runs (MuRIL 76.93 vs the 10-epoch baseline 81.71; the five-epoch budget also contributes). Increasing `max_len` to 96 is mixed (MuRIL 78.05, mBERT 80.22, Distil-mBERT 80.92 at five epochs) and does not beat the full-budget baseline. The batch-8 run reaches 79.86 at five epochs, above the length-96 run (78.05) but below the 10-epoch baseline. The frozen configuration is sensitive to the learning rate, and the obvious “more context” lever does not help within a short budget.

**Table 8. Transformer hyperparameter sensitivity (validation macro-F1, %, seed 42, five fixed epochs).**

| Model | Config | LR | Max len | Batch | Eff. batch | Best ep. | Val macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| MuRIL LoRA | baseline (main) | 4×10⁻⁴ | 64 | 16 | 32 | 8 | **81.71** |
| MuRIL LoRA | lr 2×10⁻⁴ | 2×10⁻⁴ | 64 | 16 | 32 | 5 | 76.93 |
| MuRIL LoRA | len 96 | 4×10⁻⁴ | 96 | 16 | 32 | 5 | 78.05 |
| MuRIL LoRA | batch 8 | 4×10⁻⁴ | 64 | 8 | 32 | 5 | 79.86 |
| mBERT LoRA | baseline (main) | 4×10⁻⁴ | 64 | 16 | 32 | 3 | **80.61** |
| mBERT LoRA | lr 2×10⁻⁴ | 2×10⁻⁴ | 64 | 16 | 32 | 5 | 79.76 |
| mBERT LoRA | len 96 | 4×10⁻⁴ | 96 | 16 | 32 | 5 | 80.22 |
| mBERT LoRA | batch 8 | 4×10⁻⁴ | 64 | 8 | 32 | 5 | 79.71 |
| Distil-mBERT | baseline (main) | 4×10⁻⁴ | 64 | 16 | 32 | 6 | **81.09** |
| Distil-mBERT | lr 2×10⁻⁴ | 2×10⁻⁴ | 64 | 16 | 32 | 5 | 80.08 |
| Distil-mBERT | len 96 | 4×10⁻⁴ | 96 | 16 | 32 | 5 | 80.92 |
| Distil-mBERT | batch 8 | 4×10⁻⁴ | 64 | 8 | 32 | 5 | 80.97 |

![Figure 9: Hyperparameter sensitivity (validation macro-F1, seed 42, five fixed epochs). Lowering the learning rate hurts; length 96 and batch 8 do not beat the full-budget baseline.](figures/08_sensitivity.png)

## 6.5 Sequence Length and Truncation

At `max_len=64`, only 3.47% of test examples are truncated. The truncation rate falls to 0.17% at 96 and 0.14% at 128; at 32 it is 15.21% and at 48 it is 7.60%. The content-subword length distribution (excluding the two special tokens) has mean 17.9, 90th percentile 40, 95th percentile 56, and maximum 137. A length-96 run is therefore a defensible next experiment—it removes almost all truncation at modest cost—but the sensitivity study does not show a length-96 gain within a five-epoch budget.

**Table 9. Truncation at different maximum sequence lengths (MuRIL tokenizer, test split).** Counts include the two special tokens.

| Max length | Truncated test examples (%) |
|---:|---:|
| 32 | 15.21 |
| 48 | 7.60 |
| 64 | 3.47 |
| 96 | 0.17 |
| 128 | 0.14 |

## 6.6 Selective Prediction (RQ8)

At 50% coverage, full fine-tuning reaches 95.76 macro-F1 and logistic regression 95.74; at 80%, 90.53 and 89.34; at 90%, 87.66 and 86.34. Both curves are monotone: abstaining on the least-confident 20% raises macro-F1 by 6.4–7.2 points. The gap between the two models remains small at every coverage level, so selective prediction improves reliability but does not separate the capacity story. A 50%-coverage deployment can operate at about 96% macro-F1 with a simple confidence threshold, using either model.

![Figure 10: Risk–coverage curves. Abstaining on the least-confident predictions raises macro-F1 monotonically.](figures/05_risk_coverage.png)

---

# 7. Discussion: Why Do Linear Models Match Transformers?

## 7.1 The Direct Answer

Five mechanisms, each supported by a separate experiment, explain the result. The direct answer is that the best single-view model is the calibrated linear union (83.63 macro-F1, 379 K coefficients, 4.6 s end-to-end), while the 238 M full-FT reaches 83.36 after 53.9 min; the linear model is nominally ahead and the gap is not significant.

1. **The task is local and lexical.** Comments have median 8 tokens and median 47 characters. Polarity is carried by a small set of high-frequency cues and intensifiers (*bagundi*, *chala*, *keka*, *waste*, *bore*). A linear model over word unigrams learns these cues directly; a transformer must learn them through many layers and a much larger hypothesis space.
2. **Character n-grams absorb orthographic variation.** Romanized Telugu has many spellings for the same word (*bagundi*, *bavundhi*, *baagundhi*). Character 2–5-grams share substrings across variants; WordPiece maps them to different subword sequences. The feature ablation (char 82.76, union 83.24) and the tokenizer experiment (78.63 → 79.45) quantify both sides of this mechanism.
3. **Small data favours high bias.** With 27,670 training rows and a noisy label floor, the variance of a 238 M-parameter model is a liability. Full fine-tuning’s training loss falls from 0.886 to 0.242 while validation peaks at epoch 4 (82.55) and never recovers—textbook overfitting. LoRA reduces variance but also reduces capacity to 0.59 M parameters, and it lands at 81.65.
4. **Pretraining register mismatch.** MuRIL’s Romanized pretraining is formal and rule-based (Wikipedia and transliterated corpora), whereas Tenglish comments are informal and phonetically spelled. Full fine-tuning can shift the representation; a low-rank update cannot shift it enough. The 1.71-point LoRA/full-FT gap is the price of that mismatch.
5. **More parameters are redundant once the cues are represented.** The capacity–accuracy correlation is weak (ρ = 0.306) and non-significant. The 279 M-parameter XLM-R LoRA (81.03) is below the 238 M-parameter MuRIL LoRA (81.65), and both are below a 50.7 K-coefficient LogReg (82.93) and a 379 K-coefficient linear union (83.63). Capacity is not the bottleneck.

## 7.2 A Linguistic Reading: The Matrix-Language Frame

Myers-Scotton’s Matrix Language Frame (MLF) model [29] distinguishes the matrix language, which supplies the morphosyntactic frame, from embedded-language islands. In Tenglish, Telugu typically supplies the frame and English supplies content islands, while sentiment is carried by a small set of content morphemes (*bagundi* “good”, *waste*, *keka* “superb”) and system morphemes (*-ledu* “not”, *kaadhu* “is not”, *kani* “but”). These cues are morpheme-local and largely order-insensitive: their polarity does not depend on long-range syntactic composition. This is exactly the regime in which a unigram/character model has the right inductive bias, and it is consistent with the finding that word bigrams do not reliably improve the linear model. MLF is a grammatical theory, not a theory of model capacity; it does not by itself explain why a 238 M-parameter model fails to beat a linear one. It does explain why the linear model’s hypothesis space is well matched to the cue structure, which is the first half of the explanation.

## 7.3 The Capacity Illusion

We use the term *capacity illusion* for the inferential error of attributing a leaderboard gap to model capacity when the gap is produced by leakage, tokenization, or tuning. In our matrix, the raw capacity ordering no longer survives: the 379 K-coefficient linear union is nominally ahead of the 238 M-parameter full-FT, and the reported 5% accuracy improvement collapses to a non-significant 0.27-point macro-F1 difference once (i) the split is group-disjoint, (ii) tokenization is tested as a separate factor rather than confounded with architecture, and (iii) the linear baseline is tuned with the same search budget. The illusion is not that transformers are bad; it is that the comparison was not controlled. A benchmark that reports only the winning transformer, without a tuned linear baseline and a leakage audit, cannot distinguish “the model learned the task” from “the split leaked the task”.

## 7.4 Relation to Published Claims

The published transformer claim reports a 5% accuracy and 6% F1 improvement over baselines on CMTE-NITANP. Our leak-free comparison does not reproduce a gap of that size. There are three non-exclusive explanations: their split may differ from ours; their baseline may be untuned; or their transformer may use a different adaptation regime. We cannot adjudicate without their exact split and baseline numbers, which are not available in the accessible record. What we can say is that on a split with zero exact Jaccard ≥ 0.8 train–test pairs, with a tuned linear baseline and a frozen hyperparameter protocol, the best transformer is 0.27 points behind the best linear model (and 0.43 points ahead of the word-unigram LogReg); neither difference is significant. Any claim of a 5–6% advantage should therefore be accompanied by a split audit and a tuned linear baseline.

## 7.5 Implications

- Always report a tuned TF-IDF linear baseline. It is fast, strong, and calibrated; on this task it is within noise of the best transformer.
- Audit the split with exact Jaccard matching, not only MinHash. MinHash flagged 203 test rows (5.87%); the exact audit found 345 train–test pairs at or above 0.8.
- Prefer character n-grams or a word+character union when spelling is variable; word bigrams can hurt.
- If using a transformer, prefer full fine-tuning over low-rank adaptation when the domain mismatch is large and the GPU allows it; LoRA is a compute compromise, not a free lunch.
- Report calibration and robustness alongside accuracy. The best single-view model also has the smallest combined-noise degradation here (Δ = 10.19), but it is not the best-calibrated; calibration remains a separate axis.

## 7.6 Threats to Validity

- **Internal validity:** Split B uses a specific clustering threshold (0.8) and word 3-gram shingles. The audit is stable across 0.7–0.9, but a different shingle definition could produce a different partition. The label-noise floor (115 conflicting clusters, 314 rows) is small but non-zero and affects all models equally.
- **External validity:** The study covers one language pair and one corpus. CMTE-NITANP is a single social-media comment corpus; lexical cues and spelling conventions may differ on other platforms. CMTET has a different label scheme and domain, so it is not a clean domain-shift estimate.
- **Construct validity:** Macro-F1 treats the three classes equally, which is appropriate for the near-balanced test set but may not reflect deployment costs.
- **Statistical validity:** Full fine-tuning has one seed; its interval is a test-set bootstrap interval, not a seed-to-seed interval. A three-seed full-fine-tuning run is the main missing experiment.
- **Compute validity:** The 4 GB GPU forces batch 8 with accumulation and 8-bit AdamW for full fine-tuning. A larger GPU might change the optimal batch and learning rate.

---

# 8. Concluding Remarks

We presented a leakage-audited benchmark of 21 configurations and 85 runs for Romanized Telugu–English sentiment on CMTE-NITANP. On a group-disjoint split with zero exact Jaccard ≥ 0.8 train–test pairs, the best single-view model is a 379 K-coefficient calibrated LinearSVC on the word+character union, at **83.63 macro-F1 and 83.64 accuracy in 4.6 s end-to-end**; a post-hoc multi-view variant reaches **83.92/83.93**. A 238 M-parameter fully fine-tuned MuRIL reaches **83.36/83.38** after 53.9 min; the 0.27-point difference is not statistically significant. A post-hoc six-model heterogeneous ensemble reaches **85.00/85.01** (temperature-calibrated 85.02/85.03), but strict validation-only selection gives 84.77 and a validation-fitted simplex gives 84.33; the ensemble is genuine but selection-sensitive.

Parameter count does not predict accuracy, LoRA adapters trail the linear model, word bigrams do not reliably improve the linear model, and WordPiece tokenization does not close the gap. Under orthographic perturbation, the calibrated linear union is the most robust model under combined noise (Δ = 10.19 points), while vowel elongation remains the dominant failure mode for every model. Zero-shot transfer to CMTET loses 17.54–25.70 macro-F1 points.

The public data therefore supports a **post-hoc** frontier of about **85.0**. A 90% result is not credible without leakage or a different dataset/label definition; the realistic label-noise ceiling is **~86.0–86.8%**.

---

# 9. Future Work

1. **Multi-seed full fine-tuning.** Run MuRIL full-FT with seeds 43 and 44 and report mean ± standard deviation; the current single-seed interval is a test-set interval, not a cross-seed interval.
2. **Model soup and checkpoint averaging.** Average the full-FT best and last checkpoints; this is a low-cost way to improve the strongest transformer.
3. **English encoders.** BERT-base and Electra-base are strong on Romanized Tenglish; a controlled comparison with MuRIL on the same split is a useful follow-up.
4. **Hybrid ensemble with validation-fitted, constrained weights.** Use a simplex-constrained blend and 5-fold validation to avoid the selection sensitivity of the current 85.0 result.
5. **Calibration and abstention.** Temperature-scale the ensemble and report risk–coverage; the current transformers are over-confident.
6. **Larger GPU and longer context.** A 16 GB GPU would allow XLM-R full fine-tuning and max_len 96/128 without the 4 GB compromises.
7. **Deployment study.** Evaluate the linear/ensemble frontier under latency, memory, and calibration constraints for a moderation or review-analytics deployment.

---

# 10. References

1. U. R. Rayala, K. Seshadri, and N. B. Sristy, “Sentiment analysis of code-mixed Telugu–English data leveraging syllable and word embeddings,” *ACM Trans. Asian Low-Resour. Lang. Inf. Process.*, vol. 22, no. 10, pp. 1–30, 2023, doi: 10.1145/3620670.
2. U. R. Rayala and K. Seshadri, “Sentiment analysis of code-mixed Telugu–English text using transformers,” *J. Comput. Social Sci.*, vol. 9, art. 2, 2026, doi: 10.1007/s42001-025-00434-3.
3. S. S. V. Kusampudi, P. Sathineni, and R. Mamidi, “Sentiment analysis in code-mixed Telugu–English text with unsupervised data normalization,” in *Proc. RANLP*, 2021, pp. 753–760.
4. B. R. Chakravarthi, R. Priyadharshini, V. Muralidaran, N. Jose, S. Suryawanshi, E. Sherly, and J. P. McCrae, “DravidianCodeMix: Sentiment analysis and offensive language identification dataset for Dravidian languages in code-mixed text,” *Lang. Resour. Eval.*, 2022, doi: 10.1007/s10579-022-09583-7.
5. P. Patwa, G. Aguilar, S. Kar, S. Pandey, S. PYKL, B. Gambäck, T. Chakraborty, T. Solorio, and A. Das, “SemEval-2020 Task 9: Overview of sentiment analysis of code-mixed tweets,” in *Proc. SemEval*, 2020, pp. 774–790.
6. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
7. A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” in *Proc. ACL*, 2020, pp. 8440–8451.
8. S. Khanuja et al., “MuRIL: Multilingual representations for Indian languages,” *arXiv:2103.10730*, 2021.
9. D. Kakwani, A. Kunchukuttan, S. Goyal, G. N.C., A. Bhattacharyya, M. M. Khapra, and P. Kumar, “IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages,” in *Findings of EMNLP*, 2020, pp. 4948–4961.
10. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.
11. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in *Proc. NeurIPS*, 2023.
12. T. Wolf et al., “Transformers: State-of-the-art natural language processing,” in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.
13. F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
14. L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.
15. J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.
16. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. ACM SIGKDD*, 2016, pp. 785–794.
17. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in *Proc. NeurIPS*, 2017, pp. 3146–3154.
18. C. Cortes and V. Vapnik, “Support-vector networks,” *Mach. Learn.*, vol. 20, no. 3, pp. 273–297, 1995.
19. T. Cover and P. Hart, “Nearest neighbor pattern classification,” *IEEE Trans. Inf. Theory*, vol. 13, no. 1, pp. 21–27, 1967.
20. C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge, U.K.: Cambridge Univ. Press, 2008.
21. G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513–523, 1988.
22. W. B. Cavnar and J. M. Trenkle, “N-gram-based text categorization,” in *Proc. SDAIR*, 1994, pp. 161–175.
23. S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, pp. 1–21, 2012.
24. T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998.
25. S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.
26. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.
27. C. Nadeau and Y. Bengio, “Inference for the generalization error,” *Mach. Learn.*, vol. 52, no. 3, pp. 239–281, 2003.
28. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. ICML*, 2017, pp. 1321–1330.
29. C. Myers-Scotton, *Contact Linguistics: Bilingual Encounters and Grammatical Outcomes*. Oxford, U.K.: Oxford Univ. Press, 2002.

---

# Appendix: Implementation Artifacts

## A.1 Repository Layout

| Path | Contents |
|---|---|
| `datasets/raw/CMTE-NITANP/34k_Code_Mixed_Tel-Eng.csv` | Public CMTE-NITANP release (34,588 rows) |
| `datasets/splits/CMTE-NITANP/` | Split A/B JSONL, clustered data, split report |
| `Implementation/benchmark/` | Benchmark runners, predictions, metrics, logs, reports |
| `Implementation/benchmark/results/` | `master_results.csv`, `seed_summary.csv`, `feature_ablation.csv`, `stats/pairwise_vs_logreg.csv` |
| `Implementation/benchmark/reports/` | `REPORT.md`, aggregate tables, plots |
| `Implementation/benchmark/ensemble/` | Post-hoc linear models, ensembles, hybrid, results |
| `architecture/paper/` | IEEE manuscript, UROP report, figures |
| `models/` | Downloaded BERT/Electra weights |

## A.2 Main Commands

Representative commands; adjust paths and flags to your checkout.

```bash
# Split audit
python Implementation/benchmark/audit/split_audit.py --dataset CMTE-NITANP

# Classical/transformer benchmark
python Implementation/benchmark/run_queue.py

# Aggregate results
python Implementation/benchmark/aggregate.py

# Post-hoc ensemble audit
python Implementation/benchmark/ensemble/audit_ensemble.py
python Implementation/benchmark/ensemble/blend.py
python Implementation/benchmark/ensemble/temperature_blend.py

```

## A.3 Environment

- GPU: NVIDIA RTX 2050, 4 GB (3.87 GB usable)
- PyTorch with CUDA, Hugging Face Transformers [12], PEFT, bitsandbytes, scikit-learn [13]
- TeX: Tectonic 0.17 for the IEEE manuscript
- Seed: 42 for search/freeze and final runs; 42–44 for LoRA seeds; 42–46 for classical/deep seeds

---

*End of report.*
