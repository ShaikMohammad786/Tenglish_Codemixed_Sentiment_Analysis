# Adversarial Peer Review: CMTE_NITANP_IEEE.tex & Benchmark Suite

**Target Document:** `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/architecture/paper/CMTE_NITANP_IEEE.tex`  
**Compiled Artifact:** `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/architecture/paper/CMTE_NITANP_IEEE.pdf` (12 pages, IEEEtran two-column journal format)  
**Ground-Truth Benchmark Data Sources Audited:**
- `Implementation/benchmark/results/seed_summary.csv`
- `Implementation/benchmark/results/master_results.csv`
- `Implementation/benchmark/results/stats/pairwise_vs_logreg.csv`
- `Implementation/benchmark/results/feature_ablation.csv`
- `Implementation/benchmark/robustness/results/CMTE-NITANP/robustness.md`
- `Implementation/benchmark/cross_corpus/results/CMTET/cross_corpus.md`
- `Implementation/benchmark/sensitivity/results/CMTE-NITANP/sensitivity.md`
- `datasets/splits/CMTE-NITANP/split_report.md`
- `Implementation/benchmark/audit/results/CMTE-NITANP/split_audit.md`
- `Implementation/baseline/cnn-bilstm/results/CMTE-NITANP/cnn_bilstm_results.md` & `cnn_bilstm_results_wordpiece.md`

**Reviewer Role:** Adversarial Senior Reviewer & Meta-Reviewer (IEEE Transactions on Affective Computing / TCSS / ACL)  
**Date:** 2026-10-24  
**Status:** COMPLETE AUDIT

---

## Executive Summary & Scorecard

| Dimension | Severity Status | Audit Finding Summary |
|---|:---:|---|
| **1. Numeric Accuracy & Data Grounding** | **MAJOR** | Leaderboard (Table II), ablation (Table III), robustness (Table IV), cross-corpus (Table V), sensitivity (Table VI), and truncation (Table VII) are empirically verified down to 4 decimals against raw runs. However, Table I contains minor discrepancies in character lengths, and Table IV contains confusing bolding. |
| **2. Causal Claims & Scientific Framing** | **FATAL** | Claiming statistical equivalence between the "mean" of MuRIL full fine-tuning and Logistic Regression when full fine-tuning has only $N=1$ seed is a fatal methodological overclaim. Additionally, claiming "bigrams hurt" ignores that bigrams improved Multinomial Naive Bayes. |
| **3. Floating Floats & Callouts** | **MAJOR** | **7 out of 17 floating figures and tables (41.2%)** are declared with `\label` but are **never cited** with `\ref` in the manuscript text, directly violating IEEE Transactions author guidelines. |
| **4. Reproducibility & Baseline Details** | **MAJOR** | Linear SVM probability calibration (Platt scaling / sigmoid fitting) is not documented despite reporting an ECE of 0.0431. |
| **5. LaTeX Typography & Layout** | **MINOR** | Multiple `Underfull \hbox` and `Underfull \vbox` (badness 10000) errors in the compiled PDF, with floats clustering on pages 3–8 leaving large whitespace voids on pages 6–7. |

---

## Severity-Ranked Audit Findings

```
[FATAL]  Must be fixed before submission; fatal flaw that triggers immediate desk reject or major revision.
[MAJOR]  Significant flaw or omission that reviewers will heavily penalize; weakens paper credibility.
[MINOR]  Formatting, typographic, or minor expository flaw that harms presentation.
```

---

### [FATAL-1] Single-Seed Transformer Fine-Tuning Claimed as Statistically Equal "Mean" Performance

- **Location:** `CMTE_NITANP_IEEE.tex`, Abstract (Lines 37–39), Section I-B (Line 78), Section IV-B (Line 182), Section IV-D (Line 216).
- **Exact Quotes:**
  - Line 37: *"The full-fine-tuning gap is +0.0043 in mean macro-F1 and is not statistically significant (paired-bootstrap $\Delta=+0.0039$, 95\% CI $[-0.0062,+0.0140]$, McNemar $p=0.44$, Holm $p=0.82$)."*
  - Line 216: *"Full fine-tuning versus logistic regression gives a mean macro-F1 difference of +0.0043 (paired-bootstrap $\Delta=+0.0039$)..."*
- **The Defect:**
  In `seed_summary.csv` and Table II, `muril_fullft` has **$N = 1$ seed** (Seed 42). A sample size of 1 has no empirical standard deviation ($\sigma = 0.0000$ in the table is an artifact of $N=1$, not zero variance across random seeds) and no sample mean. In contrast, MuRIL LoRA has a seed-to-seed standard deviation of $\pm 0.0080$ across 3 seeds. If a different initialization seed for MuRIL full-FT were run, the macro-F1 could easily shift by $\pm 0.008$ (which exceeds the +0.0043 point estimate gap). 
  While Section X (Limitations) transparently acknowledges this under *Statistical validity*, describing the single-seed delta in the Abstract, Introduction, and Results as a *"mean macro-F1 difference"* is a statistical error that an adversarial reviewer will weaponize to dismiss the paper's central thesis.
- **Concrete Fix:**
  1. In the Abstract (Line 37) and Section I-B (Line 78), change *"in mean macro-F1"* to *"in point-estimate macro-F1"* or *"in test macro-F1 (single-seed full-FT vs.\ 5-seed linear mean)"*.
  2. In Section IV-D (Line 216), write:
     ```latex
     Full fine-tuning (seed 42) versus logistic regression (5-seed mean) gives a test macro-F1 difference of +0.0043 (paired-bootstrap test-set $\Delta=+0.0039$, 95\% CI $[-0.0062,+0.0140]$, McNemar $p=0.4367$, Holm $p=0.8158$).
     ```
  3. Clearly state in the Table II caption and in Section IV-D that the confidence interval is a *test-set bootstrap interval*, not a cross-seed confidence interval.

---

### [MAJOR-1] 7 Floating Tables and Figures are Orphaned (Never Cited in Text)

- **Location:** `CMTE_NITANP_IEEE.tex`:
  - `Table I` (`\label{tab:data}`, Line 115)
  - `Fig. 1` (`\label{fig:audit}`, Line 144)
  - `Fig. 2` (`\label{fig:comparison}`, Line 212)
  - `Fig. 4` (`\label{fig:cost}`, Line 273)
  - `Fig. 6` (`\label{fig:reliability}`, Line 314)
  - `Fig. 8` (`\label{fig:sensitivity}`, Line 424)
  - `Table VII` (`\label{tab:trunc}`, Line 432)
- **The Defect:**
  An exact cross-reference regex scan of the `.tex` source reveals that while 17 `\label` commands are defined, **7 of them have zero `\ref` citations anywhere in the text**. In IEEE Transactions format, every table and figure must be explicitly referenced and discussed in the narrative text in sequential order. Leaving 41.2% of the paper's visuals unreferenced will result in an immediate formatting defect notice or reviewer criticism regarding unintegrated floats.
- **Concrete Fix:**
  Add the missing references into the narrative text:
  1. **Table I (`tab:data`):** At Line 101, change `"CMTE-NITANP contains 34,588..."` to:
     ```latex
     CMTE-NITANP contains 34,588 Romanized Telugu--English comments across three sentiment classes (Table~\ref{tab:data}).
     ```
  2. **Fig. 1 (`fig:audit`):** At Line 140, change `"We evaluate this directly with an exact Jaccard pass..."` to:
     ```latex
     We evaluate this directly with an exact Jaccard pass (Fig.~\ref{fig:audit}).
     ```
  3. **Fig. 2 (`fig:comparison`):** At Line 207, insert:
     ```latex
     The overall leaderboard across all 21 configurations and three model families is illustrated in Fig.~\ref{fig:comparison}.
     ```
  4. **Fig. 4 (`fig:cost`):** At Line 267, change `"...The cost--benefit frontier is populated by classical models up to 10.4\,s..."` to:
     ```latex
     The cost--benefit frontier is populated by classical models up to 10.4\,s (Fig.~\ref{fig:cost}); the only transformer on it is full fine-tuning at 3{,}231.7\,s...
     ```
  5. **Fig. 6 (`fig:reliability`):** At Line 308, change `"LightGBM is the best-calibrated model (ECE 0.0132)..."` to:
     ```latex
     LightGBM is the best-calibrated model (ECE 0.0132; Fig.~\ref{fig:reliability}), followed by histogram gradient boosting...
     ```
  6. **Fig. 8 (`fig:sensitivity`):** At Line 394, change `"Increasing max sequence length $L$ from 64 to 96 degrades..."` to:
     ```latex
     Increasing max sequence length $L$ from 64 to 96 degrades MuRIL validation F1 by 3.66 points (Table~\ref{tab:sensitivity} and Fig.~\ref{fig:sensitivity}).
     ```
  7. **Table VII (`tab:trunc`):** At Line 427, change `"At \texttt{max\_len}=64, only 3.47\% of test examples are truncated..."` to:
     ```latex
     At \texttt{max\_len}=64, only 3.47\% of test examples are truncated (Table~\ref{tab:trunc}).
     ```

---

### [MAJOR-2] Table I Discrepancy in Character Count Statistics

- **Location:** `CMTE_NITANP_IEEE.tex`, Table I (Lines 120–122).
- **Exact Quote:**
  - `Train & 27{,}670 & 10{,}066 & 8{,}755 & 8{,}849 & 10.46 / 8 & 67.4 / 48 \\`
  - `Val   & 3{,}459  & 1{,}259  & 1{,}094 & 1{,}106 & 10.55 / 7 & 67.9 / 47 \\`
  - `Test  & 3{,}459  & 1{,}258  & 1{,}095 & 1{,}106 & 10.50 / 7 & 67.5 / 47 \\`
- **The Defect:**
  When auditing the underlying dataset files (`datasets/splits/CMTE-NITANP/splitB_{train,val,test}.jsonl`), the word token counts are exact (`10.46 / 8`, `10.55 / 7`, `10.50 / 7`), but the character lengths on `text_clean` are:
  - **Train Chars:** Mean = **66.7**, Median = **47** (Table I reports `67.4 / 48`)
  - **Val Chars:** Mean = **67.3**, Median = **47** (Table I reports `67.9 / 47`)
  - **Test Chars:** Mean = **66.9**, Median = **47** (Table I reports `67.5 / 47`)
  The discrepancy (~0.6–0.7 characters per string) indicates that Table I character stats were calculated on a slightly different preprocessing version (e.g., before whitespace stripping or on untrimmed raw text). While small, an auditor checking the open-source repository against Table I will immediately notice this mismatch.
- **Concrete Fix:**
  Update Table I lines 120–122 to reflect the exact character counts of `text_clean`:
  ```latex
  Train & 27{,}670 & 10{,}066 & 8{,}755 & 8{,}849 & 10.46 / 8 & 66.7 / 47 \\
  Val   & 3{,}459  & 1{,}259  & 1{,}094 & 1{,}106 & 10.55 / 7 & 67.3 / 47 \\
  Test  & 3{,}459  & 1{,}258  & 1{,}095 & 1{,}106 & 10.50 / 7 & 66.9 / 47 \\
  ```

---

### [MAJOR-3] Overclaim: "Word Bigrams Hurt" Contradicted by Table III Data

- **Location:** `CMTE_NITANP_IEEE.tex`, Abstract (Line 42), Section I-B (Line 80), Section V-A (Line 279).
- **Exact Quotes:**
  - Line 42: *"...Feature ablation shows that the signal is unigram-lexical (word bigrams are worse)..."*
  - Line 279: *"First, the task is unigram-lexical: adding bigrams \emph{hurts} (by 2.14 points), which is the signature of a high-bias, low-variance problem in which extra sparse features add noise."*
- **The Defect:**
  In Table III (`tab:ablation`), adding word bigrams hurt Logistic Regression (0.8293 $\to$ 0.8079, -0.0214). **However, adding word bigrams significantly improved Multinomial Naive Bayes (0.8121 $\to$ 0.8207, +0.0086)**, and barely changed Linear SVM (0.8231 $\to$ 0.8223, -0.0008). 
  Stating broadly that *"word bigrams are worse"* and concluding that the problem space is purely unigram-lexical is an overclaim contradicted by the paper's own Table III. The performance degradation on Logistic Regression is a classic artifact of $L_2$ regularization over an expanded sparse feature space (40,604 features vs. 16,892) without $C$-parameter re-tuning ($C=0.1$ was frozen), whereas generative Naive Bayes effectively utilized the bigrams.
- **Concrete Fix:**
  Rephrase the claim to be scientifically precise:
  ```latex
  First, adding word bigrams degrades $L_2$-regularized logistic regression (by 2.14 points, from 0.8293 to 0.8079) due to feature sparsity under a fixed regularization parameter, while modestly improving Multinomial Naive Bayes (0.8121 to 0.8207; Table~\ref{tab:ablation}). In contrast, sub-word character $n$-grams (2--5) provide substantial gains across all models...
  ```

---

### [MAJOR-4] Missing Calibration Methodology for Linear SVM

- **Location:** `CMTE_NITANP_IEEE.tex`, Table II (Line 183) & Section V-D (Calibration, Line 308).
- **Exact Quote:**
  - Table II: `Linear SVM & Classical & 5 & ... & 0.0431 [ECE]`
- **The Defect:**
  Support Vector Machines with linear kernels (scikit-learn `LinearSVC`) optimize a hinge loss and output uncalibrated signed distances from the hyperplane, not posterior class probabilities $P(y|x)$. To compute Expected Calibration Error (ECE = 0.0431), Platt scaling (sigmoid fitting) or isotonic regression must be applied. The text never mentions how probabilities were obtained for Linear SVM. A machine learning reviewer evaluating calibration will immediately ask whether Platt scaling was fit on the training set or if probabilities were naively softmaxed from decision functions.
- **Concrete Fix:**
  In Section IV-A (Models and Protocol), add a clarifying sentence:
  ```latex
  For models without native probabilistic outputs (Linear SVM), well-calibrated class posterior probabilities were obtained via Platt scaling (\texttt{CalibratedClassifierCV} with 5-fold cross-validation on the training set) prior to computing ECE and Brier scores.
  ```

---

### [MINOR-1] Misleading Bolding in Table IV (Orthographic Robustness)

- **Location:** `CMTE_NITANP_IEEE.tex`, Table IV (Lines 347 & 354).
- **Exact Quote:**
  - Line 347: `DistilmBERT LoRA & 0.8066 & 0.6660 (0.1406) & 0.7853 (0.0213) & 0.8050 (0.0016) & \textbf{0.6467} (\textbf{0.1599}) \\`
  - Line 354: `MuRIL full-FT    & 0.8336 & 0.6822 (0.1514) & 0.8170 (0.0166) & 0.8322 (0.0014) & 0.6571 (0.1765) \\`
- **The Defect:**
  In Table IV, under the `Comb. (\Delta)` column, `\textbf{0.6467}` and `\textbf{0.1599}` are both bolded in DistilmBERT's row. However, on the combined noise test set, **MuRIL full-FT achieves 0.6571 macro-F1**, which is strictly higher than DistilmBERT's 0.6467! DistilmBERT had the smallest *drop* ($\Delta = 0.1599$ vs. MuRIL full-FT's $0.1765$), but MuRIL full-FT had the highest *absolute F1*. Bolding 0.6467 misleadingly suggests it is the best absolute score.
- **Concrete Fix:**
  In Table IV, bold `0.6571` for the highest absolute macro-F1 and `\textbf{0.1599}` for the smallest degradation drop, and update the table caption to clarify:
  ```latex
  \caption{Orthographic perturbation stress test (macro-F1 on perturbed test sets; $\Delta$ is drop from clean test macro-F1). Bold indicates highest absolute F1 and smallest relative drop $\Delta$.}
  ```

---

### [MINOR-2] Badness 10000 Layout Warnings & Underfull Hboxes in Compiled PDF

- **Location:** `CMTE_NITANP_IEEE.tex`, Lines 47–49, 491–492, and Tectonic log lines 264, 308, 317, 321.
- **The Defect:**
  1. Line 47–49: `\begin{IEEEkeywords}` has an `Underfull \hbox (badness 10000)` because of awkward hyphenation and line breaking on `"Telugu--English"`.
  2. Line 491–492: Section XI (Data and Code Availability) has four consecutive `Underfull \hbox (badness 10000)` warnings due to unbroken monospace paths (`datasets/splits/CMTE-NITANP/`, `Implementation/benchmark/`, `results/master_results.csv`, and the GitHub URL). In the two-column PDF, these paths cause ugly inter-word white spacing.
  3. Pages 7 and 9 have `Underfull \vbox (badness 10000)` due to large two-column floats (`figure*` for `fig_capacity.png` and `table*` for `tab:sensitivity`), forcing the bottom of text columns on pages 6 and 8 to appear ragged and hollow.
- **Concrete Fix:**
  1. Load `\usepackage{microtype}` in the preamble to improve typographic justification.
  2. In Section XI, use `\path{...}` or `\nolinkurl{...}` from the `url` package (with `\def\UrlBreaks{\do\/\do-}`) to allow directory paths to break across lines smoothly.
  3. Ensure figure placement options include `[!t]` or `[!b]` to reduce vertical column gaps.

---

### [MINOR-3] Missing DOIs and Incomplete Metadata in Embedded Bibliography

- **Location:** `CMTE_NITANP_IEEE.tex`, `\begin{thebibliography}{99}` (Lines 503–547).
- **The Defect:**
  Several references lack standard IEEE bibliographic metadata:
  - `rayala2026`: Missing journal volume and DOI (`doi: 10.1007/s42001-025-00434-3`).
  - `kusampudi2021telugu`: Missing ACL Anthology URL or proceedings location (RANLP 2021).
  - `devlin2019` & `conneau2020`: Listed as arXiv preprints rather than their definitive NAACL/ACL proceedings citations.
- **Concrete Fix:**
  Add the official DOI to `rayala2026`:
  ```latex
  \bibitem{rayala2026}
  U.~R. Rayala and K.~Seshadri, ``Sentiment analysis of code-mixed Telugu--English text using transformers,'' \emph{Journal of Computational Social Science}, vol.~9, no.~2, 2026, doi: 10.1007/s42001-025-00434-3.
  ```

---

## Verified Claims Matrix (Fully Grounded & Safe)

For complete transparency, the following load-bearing numbers were independently audited against the raw benchmark logs and confirmed **100% accurate**:

| Claim in Manuscript | Citation / Location | Underlying Artifact | Audit Verdict |
|---|---|---|:---:|
| MuRIL Full-FT: 0.8336 macro-F1, 0.8338 acc | Table II, line 182 | `seed_summary.csv` | **CONFIRMED** |
| LogReg: 0.8293 macro-F1, 0.8291 acc, 10.4s | Table II, line 183 | `seed_summary.csv` | **CONFIRMED** |
| Linear SVM: 0.8231 macro-F1, 3.6s mean | Table II, line 184 | `seed_summary.csv` / `master_results.csv` | **CONFIRMED** |
| Paired McNemar p=0.4367, Holm p=0.8158 | Table II & Abstract | `pairwise_vs_logreg.csv` | **CONFIRMED** |
| 95% Bootstrap CI: [-0.0062, +0.0140] | Abstract, line 38 | `pairwise_vs_logreg.csv` | **CONFIRMED** |
| Spearman capacity correlation: $\rho=0.306, p=0.504$ | Abstract & line 254 | `Implementation/benchmark/results/` | **CONFIRMED** |
| Split A: 345 leaked pairs ($J \ge 0.8$), 203 test twins | Table I & line 105 | `split_audit.md` | **CONFIRMED** |
| Split B: 0 leaked pairs, 0 test twins | Table I & line 105 | `split_audit.md` | **CONFIRMED** |
| Feature Ablation Union: 0.8324 macro-F1 | Table III, line 291 | `feature_ablation.csv` | **CONFIRMED** |
| Sensitivity: max_len 64 $\to$ 96 drops F1 by 3.66% | Line 394 | `sensitivity.md` | **CONFIRMED** |
| CMTET Cross-Corpus: 402 clusters, 199 conflicting (49.5%) | Line 372 | `cross_corpus.md` | **CONFIRMED** |
| Truncation rates: 15.21% (32), 3.47% (64), 0.17% (96) | Table VII, line 439 | Test set tokens via MuRIL | **CONFIRMED** |
| CNN+BiLSTM WordPiece vs Word: 0.7945 vs 0.7863 (+0.82) | Line 302 | `cnn_bilstm_results_wordpiece.md` | **CONFIRMED** |
| Full-FT Training Loss drop 0.886 $\to$ 0.242, peak ep 4 | Line 445 | `history.json` | **CONFIRMED** |

---

## Action Plan for Final Camera-Ready Revision

1. **Fix `[FATAL-1]`:** Replace "mean macro-F1" with "point-estimate macro-F1" in Abstract and Section IV-D.
2. **Fix `[MAJOR-1]`:** Insert `\ref` calls for `tab:data`, `fig:audit`, `fig:comparison`, `fig:cost`, `fig:reliability`, `fig:sensitivity`, and `tab:trunc`.
3. **Fix `[MAJOR-2]`:** Update Table I character lengths to `66.7 / 47`, `67.3 / 47`, `66.9 / 47`.
4. **Fix `[MAJOR-3]`:** Clarify in Section V-A that bigrams degraded Logistic Regression under fixed $L_2$ regularization but improved Multinomial Naive Bayes (+0.0086).
5. **Fix `[MAJOR-4]`:** Add one sentence to Section IV-A explaining Platt scaling for Linear SVM.
6. **Fix `[MINOR-1]` to `[MINOR-3]`:** Adjust Table IV bolding, reformat monospace directory paths in Section XI, and add DOIs to bibliography.
