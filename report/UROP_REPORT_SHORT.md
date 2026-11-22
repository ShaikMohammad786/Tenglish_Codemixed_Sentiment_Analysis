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

I thank my mentor, **Dr. Yandrapati Prakash Babu**, for his guidance and careful reviews throughout this project. I am grateful to **SRM University–AP** and the **School of Engineering and Sciences** for the academic environment and computing resources. I also thank the authors of the CMTE-NITANP and CMTET datasets, and the open-source communities behind PyTorch, Hugging Face Transformers, and scikit-learn.

---

# Table of Contents

- [Certificate](#certificate)
- [Acknowledgements](#acknowledgements)
- [Abstract](#abstract)
- [Abbreviations](#abbreviations)
- [List of Tables](#list-of-tables)

1. Introduction
2. Literature Survey
3. Dataset and Methodology
4. Results and Discussion
5. Concluding Remarks
6. Future Work
7. References

---

# Abstract

This report compares 21 model configurations for Romanized Telugu–English sentiment on a leakage-audited split of CMTE-NITANP. The best single-view model is a calibrated LinearSVC on word and character n-grams: **83.63 macro-F1 and 83.64 accuracy in 4.6 s**. Fully fine-tuned MuRIL (238 M parameters) reaches **83.36/83.38** after 53.9 min; the 0.27-point difference is not statistically significant (McNemar p = 0.659). Parameter count does not predict accuracy (Spearman ρ = 0.306). A post-hoc six-model ensemble reaches **85.00/85.01**, but strict validation-only selection gives 84.77, so the ensemble is selection-sensitive. On short, lexically driven code-mixed text, a linear model matches a large transformer at roughly 700× lower training cost.

**Keywords:** code-mixed sentiment analysis, Telugu–English, benchmark, data leakage, linear models, transformers.

---

# Abbreviations

| Abbreviation | Meaning |
|---|---|
| CMTE-NITANP | Code-Mixed Telugu–English corpus, NIT Andhra Pradesh |
| CMSA | Code-Mixed Sentiment Analysis |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| LoRA | Low-Rank Adaptation |
| ECE | Expected Calibration Error |
| MLF | Matrix Language Frame |

---

# List of Tables

**Table 1.** CMTE-NITANP split statistics.

**Table 2.** Key results on Split B.

---

# 1. Introduction

Code-mixing is the alternation of two or more languages within one utterance. In Telugu–English code-mixing, often written in Roman script (*Tenglish*), speakers combine English words with Romanized Telugu morphology: “movie chala bagundi” (“the movie is very good”) and “trailer keka bro” (“the trailer is superb, brother”). Sentiment analysis is difficult because the text is short, spelling varies widely, and polarity may be expressed through Telugu, English, or a hybrid construction.

CMTE-NITANP is a 34,588-comment Telugu–English sentiment corpus with negative (12,583), neutral (10,944), and positive (11,061) labels. Recent work reports that transformers improve over classical baselines by about 5% accuracy and 6% F1 [1], [2]. That claim is difficult to evaluate because the corpus contains near-duplicates and reported experiments use different splits, tokenizers, and tuning budgets. This report asks a narrow question: **on a leak-free split, with tuning held fixed, does a 238 M-parameter transformer actually beat a linear model?** We answer with 21 configurations and 85 final runs, spanning classical learners, deep networks, LoRA adapters, and full fine-tuning.

# 2. Literature Survey

CMTE-IIITH and CMTE-NITANP are the standard Telugu–English resources [1]–[3]. Shared tasks such as SemEval-2020 Task 9 popularised code-mixed sentiment [4]. Multilingual encoders—mBERT [5], XLM-R [6], MuRIL [7], and IndicBERT [8]—are the default starting point; LoRA reduces adaptation cost [9]. TF-IDF with linear classifiers remains a strong short-text baseline, but it is rarely tuned with character n-grams. Near-duplicate leakage in random splits is a known evaluation failure [10]; paired McNemar tests [11] and calibration analysis [12] are appropriate safeguards. The Matrix Language Frame (MLF) model [13] motivates the morpheme-local view of Tenglish sentiment used in the discussion.

# 3. Dataset and Methodology

CMTE-NITANP has 34,588 comments; comments are short (mean 10.46 tokens, median 8; mean 66.7 characters). We cluster the corpus with MinHash/LSH over word 3-grams and build a group-disjoint **Split B** (27,670 train / 3,459 validation / 3,459 test). Split A, a random 80/10/10 split, has 345 exact Jaccard ≥ 0.8 train–test pairs and 203 test rows with a near-duplicate in train; Split B has zero. All experiments use Split B.

We compare 11 classical learners, three deep networks, six LoRA adapters, and MuRIL full fine-tuning. Hyperparameters are selected once on validation with seed 42, frozen to JSON, and reused for final seeds; seeds vary only initialisation and data order. TF-IDF uses `min_df=2` and sublinear term frequency; the main feature is word unigrams (16,892), with word-bigram, character 2–5, and union variants in the ablation. LoRA uses r=16, α=32, learning rate 4×10⁻⁴, max_len 64, batch 16×2; full fine-tuning uses 2×10⁻⁵, 8-bit AdamW, batch 8×4, and gradient checkpointing. The primary metric is macro-F1 on the 3,459-row test set; we report accuracy, ECE, paired bootstrap intervals, and Holm-corrected McNemar tests.

**Table 1. CMTE-NITANP split statistics.**

| Split | Rows | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 27,670 | 10,066 | 8,755 | 8,849 |
| Validation | 3,459 | 1,259 | 1,094 | 1,106 |
| Test | 3,459 | 1,258 | 1,095 | 1,106 |
| **Total** | **34,588** | **12,583** | **10,944** | **11,061** |

# 4. Results and Discussion

**Table 2. Key results on Split B.** Macro-F1 and accuracy are percentages; post-hoc rows are outside the frozen 21-configuration matrix.

| Model | Macro-F1 | Accuracy | Train time |
|---|---:|---:|---:|
| MuRIL full-FT | 83.36 | 83.38 | 53.9 min |
| Logistic regression | 82.93 | 82.91 | 10.4 s |
| MuRIL LoRA | 81.65 | 81.67 | 20.8 min |
| Calibrated LinearSVC union | **83.63** | **83.64** | **4.6 s** |
| Multi-view union (post-hoc) | **83.92** | **83.93** | 12.2 s |
| Six-model ensemble (post-hoc) | **85.00** | **85.01** | — |

The best single-view model is the calibrated LinearSVC union at **83.63 macro-F1 / 83.64 accuracy in 4.6 s**. Full fine-tuned MuRIL reaches **83.36/83.38 in 53.9 min**. The 0.27-point difference is not significant (McNemar p = 0.659; 95% CI [−0.73, +1.21] percentage points), so the transformer and the linear model are statistically tied, with the linear model nominally ahead.

Parameter count does not predict macro-F1 across the seven transformer configurations (Spearman ρ = 0.306, p = 0.504). LoRA adapters (0.59 M trainable parameters) land at 81.65, below the 50.7 K-coefficient logistic regression (82.93) and the 379 K-coefficient LinearSVC union (83.63). The feature ablation shows that word bigrams are not a reliable gain (logistic regression −2.14 points; multinomial NB +0.86), while character n-grams help the margin-based and boosted models; the union is best for five of six learners. The post-hoc six-model ensemble reaches **85.00/85.01**, but strict validation-only selection gives **84.77** and a validation-fitted simplex gives **84.33**; the ensemble is genuine but selection-sensitive.

Under orthographic perturbation, vowel elongation and combined noise are the dominant failure modes. The calibrated linear union is the most robust overall: combined-noise drop **10.19 points**, versus **15.99–23.61** for the transformers. Zero-shot transfer to CMTET loses **17.54–25.70** macro-F1 points. Five mechanisms explain the linear result: the task is local and lexical; character n-grams absorb spelling variation; small data favours high bias; MuRIL’s formal pretraining mismatches informal Tenglish; and extra capacity is redundant once the cues are represented. The MLF view [13] supports the first mechanism: sentiment is carried by morpheme-local cues and is largely order-insensitive. Full fine-tuning costs roughly **700×** more compute than the best linear model for no significant gain.


# 5. Concluding Remarks

We presented a leakage-audited benchmark of 21 configurations and 85 runs for Romanized Telugu–English sentiment. On Split B, the best single-view model is a 379 K-coefficient calibrated LinearSVC union at **83.63/83.64 in 4.6 s**; a post-hoc multi-view variant reaches **83.92/83.93**. A 238 M-parameter full fine-tuned MuRIL reaches **83.36/83.38** after 53.9 min, and the difference is not significant. A post-hoc ensemble reaches **85.00/85.01**, but it is selection-sensitive. The results show that capacity is not the binding constraint on this task: a lexical linear model is statistically indistinguishable from a large transformer at a small fraction of the cost. A 90% result is not credible without leakage or a different dataset/label definition; the realistic label-noise ceiling is about **86.0–86.8%**.

# 6. Future Work

1. **Multi-seed full fine-tuning.** Run MuRIL full-FT with seeds 43 and 44 and report mean ± standard deviation.
2. **Model soup.** Average the full-FT best and last checkpoints to stabilise the strongest transformer.
3. **English encoders.** Compare BERT-base and Electra-base with MuRIL on the same split.
4. **Constrained ensemble.** Use a simplex-constrained, validation-fitted blend to reduce selection sensitivity.
5. **Calibration and abstention.** Temperature-scale the ensemble and report risk–coverage curves.

# 7. References

1. U. R. Rayala, K. Seshadri, and N. B. Sristy, “Sentiment analysis of code-mixed Telugu–English data leveraging syllable and word embeddings,” *ACM Trans. Asian Low-Resour. Lang. Inf. Process.*, vol. 22, no. 10, pp. 1–30, 2023.
2. U. R. Rayala and K. Seshadri, “Sentiment analysis of code-mixed Telugu–English text using transformers,” *J. Comput. Social Sci.*, vol. 9, art. 2, 2026.
3. S. S. V. Kusampudi, P. Sathineni, and R. Mamidi, “Sentiment analysis in code-mixed Telugu–English text with unsupervised data normalization,” in *Proc. RANLP*, 2021, pp. 753–760.
4. P. Patwa et al., “SemEval-2020 Task 9: Overview of sentiment analysis of code-mixed tweets,” in *Proc. SemEval*, 2020, pp. 774–790.
5. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
6. A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in *Proc. ACL*, 2020, pp. 8440–8451.
7. S. Khanuja et al., “MuRIL: Multilingual representations for Indian languages,” *arXiv:2103.10730*, 2021.
8. D. Kakwani et al., “IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages,” in *Findings of EMNLP*, 2020, pp. 4948–4961.
9. E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.
10. S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, pp. 1–21, 2012.
11. T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998.
12. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. ICML*, 2017, pp. 1321–1330.
13. C. Myers-Scotton, *Contact Linguistics: Bilingual Encounters and Grammatical Outcomes*. Oxford, U.K.: Oxford Univ. Press, 2002.

---

*End of report.*
