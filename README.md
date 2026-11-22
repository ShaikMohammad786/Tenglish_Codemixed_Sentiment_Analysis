# CETTE: Complementary-Error Tri-Transformer Ensemble & Dual-Regime Benchmarking for Telugu–English Code-Mixed Sentiment Analysis

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue.svg)](architecture/paper/CETTE_Conference_Paper.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Official repository for the research paper:  
**"CETTE: Complementary-Error Tri-Transformer Ensemble and Dual-Regime Benchmarking for Telugu–English Code-Mixed Sentiment Analysis"**

---

## 👥 Authors & Affiliation

- **Shaik Mohammad** (`shaik_mohammhd@srmap.edu.in`)
- **Shaik Abdullah Tayyeb** (`abdullahtayyeb_shaik@srmap.edu.in`)
- **Panditi Jayakrishna** (`jayakrishna_panditi@srmap.edu.in`)
- **Dr. Prakash Babu Yandrapati** (`prakashbabu.y@srmap.edu.in`)

*Department of Computer Science & Engineering, SRM University-AP, Amaravati, Andhra Pradesh, India.*

---

## 📌 Executive Summary

Romanized Telugu–English social media text (*Tenglish*) poses severe challenges for natural language processing: non-standard phonetic spelling variations, informal mixed syntax, subword token fragmentation, and silent near-duplicate data leakage across random splits. 

This repository provides an end-to-end, reproducible research benchmark:
1. **The Capacity Paradox Study:** An audited 21-model configuration benchmark (85 independent training runs) revealing that a fast, calibrated Linear Support Vector Classifier (LinearSVC) using character $n$-grams (379K parameters, 4.6 seconds) statistically matches a 238M-parameter fully fine-tuned MuRIL transformer (83.38% vs. 83.64%; McNemar $p = 0.659$). Across models, parameter scale exhibits a negligible correlation with accuracy (Spearman $\rho = 0.306, p = 0.504$).
2. **CETTE System:** A novel **Complementary-Error Tri-Transformer Ensemble** combining three distinct pretraining paradigms:
   - **ELECTRA-base** (Replaced Token Detection; specialized for high positive precision at 93.30%)
   - **DeBERTa-v3** (Disentangled Relative Attention; high negative recall at 94.81%)
   - **BERT-base** (Bidirectional Masked LM; stabilizing anchor)
   - Trained under **Focal-Smoothed Optimization ($\mathcal{L}_{\text{FSO}}$)** ($\gamma=1.0, \epsilon=0.05$) with temperature-calibrated Dirichlet late fusion ($r_{\text{err}} < 0.42$).
3. **Dual-Regime Benchmarking Standard:**
   - **Polar Regime ($\mathcal{P}_{\text{polar}}$: Negative vs. Positive):** Single ELECTRA reaches **88.67%** accuracy, CETTE reaches **90.15% accuracy (90.10% Macro-F1)**, and a heterogeneous super-ensemble reaches **90.84%**, advancing past the prior published benchmark (86.22%).
   - **Full 3-Class Regime ($\mathcal{P}_{\text{3-class}}$: Negative, Neutral, Positive):** Evaluated on a strictly audited, leak-free group-disjoint split (Split B: 0 leakage pairs vs. 345 pairs in random splits). CETTE attains **83.92% Macro-F1** (ECE 0.0341), while a heterogeneous super-ensemble reaches **85.02%**.
4. **Generalization & Cross-Corpus Validation:** Comprehensive orthographic stress tests (voicing alternation, vowel repetition, consonant gemination) and zero-shot cross-corpus transfer on the **CMTET** corpus (19,868 comments).

---

## 📊 Dual-Regime Empirical Results

### 1. Polar Benchmark Regime ($\mathcal{P}_{\text{polar}}$: Positive vs. Negative)

| Model Architecture | Accuracy (%) | Macro-F1 (%) | Precision (%) | Recall (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Prior Published Benchmark (Rayala & Seshadri, 2026) | 86.22 | 88.69 | — | — | Reported Polar Baseline |
| Single ELECTRA-base (with $\mathcal{L}_{\text{FSO}}$) | 88.67 | 88.51 | 89.37 | 88.67 | Single Best Backbone |
| Single DeBERTa-v3-base | 87.94 | 87.89 | 88.10 | 87.94 | High Negative Recall |
| Single BERT-base-uncased | 87.12 | 87.05 | 87.45 | 87.12 | Variance Anchor |
| **CETTE (Tri-Transformer Ensemble)** | **90.15** | **90.10** | **90.22** | **90.15** | **+3.93% over Published Baseline** |
| **Heterogeneous Super-Ensemble (Linear + CETTE)** | **90.84** | **90.79** | **90.91** | **90.84** | **Upper Empirical Bound** |

### 2. Full 3-Class Benchmark Regime ($\mathcal{P}_{\text{3-class}}$: Leak-Free Split B)

| Model Configuration | Model Family | Parameters | Train Time | Accuracy (%) | Macro-F1 (%) | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Dummy (Majority Class) | Baseline | — | < 1 s | 46.12 | 21.04 | — |
| TF-IDF Word LogReg | Classical | 45.8 K | 2.1 s | 82.94 | 82.93 | 0.0418 |
| **Calibrated LinearSVC (Word+Char Union)** | Classical | 379.1 K | **4.6 s** | **83.64** | **83.63** | 0.0372 |
| Post-Hoc Multi-View Linear (+ PLD Norm) | Classical | 752.4 K | 12.2 s | 83.93 | 83.92 | 0.0354 |
| TextCNN (Word-Level) | Deep Learning | 1.8 M | 3.4 min | 81.12 | 81.08 | 0.0582 |
| BiLSTM (Word-Level) | Deep Learning | 2.4 M | 6.8 min | 81.45 | 81.41 | 0.0541 |
| Distil-mBERT (LoRA) | Transformer | 136 M (0.6M) | 8.2 min | 80.95 | 80.92 | 0.0631 |
| XLM-RoBERTa-base (LoRA) | Transformer | 279 M (1.2M) | 14.5 min | 81.08 | 81.03 | 0.0674 |
| MuRIL-base (LoRA) | Transformer | 238 M (1.2M) | 12.1 min | 81.71 | 81.65 | 0.0612 |
| **MuRIL-base (Full Fine-Tuning)** | Transformer | 238 M | 53.9 min | 83.38 | 83.36 | 0.0589 |
| Single ELECTRA-base (with $\mathcal{L}_{\text{FSO}}$) | Transformer | 110 M | 21.4 min | 82.43 | 82.41 | 0.0512 |
| Single DeBERTa-v3-base | Transformer | 86 M | 24.1 min | 82.48 | 82.45 | 0.0498 |
| Single BERT-base-uncased | Transformer | 110 M | 19.8 min | 82.99 | 82.98 | 0.0487 |
| **CETTE (Tri-Transformer Late Fusion)** | **Ensemble** | **306 M** | **65.3 min** | **83.92** | **83.92** | **0.0341** |
| **Heterogeneous Super-Ensemble** | **Ensemble** | **307 M** | **66.1 min** | **85.02** | **85.03** | **0.0318** |

---

## 🏗️ CETTE System Architecture

```
                                      [Raw Code-Mixed Utterance]
                                    ("movie chala bagundi kani...")
                                                  │
                                                  ▼
                                   [Signal-Preserving Preprocessor]
                                   • NFC Unicode Normalization
                                   • Entity Masking (User/URL)
                                   • Emojis & Punctuation Guarded
                                   • Vowel Elongation Contraction
                                                  │
                                                  ▼
                        ┌─────────────────────────┼─────────────────────────┐
                        │                         │                         │
                        ▼                         ▼                         ▼
               [Branch A: ELECTRA]      [Branch B: DeBERTa-v3]     [Branch C: BERT-base]
               • Replaced Token Det.    • Disentangled Attention   • Bidirectional MLM
               • High Pos Precision     • High Neg Recall          • Variance Anchor
               • 105k Vocab             • 128k SPM Vocab           • 30k WordPiece
                        │                         │                         │
                        └─────────────────────────┼─────────────────────────┘
                                                  │
                                  [Focal-Smoothed Loss: L_FSO]
                            Hard-example focal (γ=1.0) + Smoothing (ε=0.05)
                                                  │
                                                  ▼
                               [Temperature-Calibrated Late Fusion]
                                    Logit Scaling: s_m / T_m
                                   Convex Dirichlet Softmax Blend
                               Pairwise Error Decorrelation (r < 0.42)
                                                  │
                                                  ▼
                                     [Dual-Regime Predictions]
                             ┌────────────────────┴────────────────────┐
                             ▼                                         ▼
                   [Regime 1: Polar P_polar]             [Regime 2: 3-Class P_3class]
                   Accuracy: 90.15% (CETTE)               Macro-F1: 83.92% (CETTE)
                   Super-Ens: 90.84%                      Super-Ens: 85.02%
```

---

## 🔍 Data Leakage Audit: Split A vs. Split B

Standard random splitting on social media datasets often suffers from near-duplicate cross-contamination. We performed an exact-Jaccard token similarity audit ($\tau \ge 0.80$):

| Split Configuration | Strategy | Leaking Train–Test Pairs | Test Rows with Train Twins | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Split A** (Standard Random) | 80/10/10 Stratified Random | **345** | **203** | ❌ Contaminated |
| **Split B** (Our Group-Disjoint) | `StratifiedGroupKFold` on Text Fingerprints | **0** | **0** | ✅ Strictly Leak-Free |

All primary benchmarks in this paper are conducted on **Split B** to prevent inflated performance metrics.

---

## 📁 Repository Structure

```
├── README.md                               <-- Root documentation & overview
├── .gitignore                              <-- Git exclusions (models, checkpoints, envs)
├── generate_paper_plots.py                 <-- Master figure generation (Vector PDF + 400 DPI PNG)
│
├── architecture/
│   ├── paper/
│   │   ├── CETTE_Conference_Paper.tex      <-- Final 10-page IEEE conference manuscript
│   │   ├── CETTE_Conference_Paper.pdf      <-- Compiled 10-page camera-ready PDF
│   │   ├── CMTE_NITANP_IEEE.tex            <-- Preserved baseline journal draft
│   │   ├── CMTE_NITANP_IEEE.pdf            <-- Preserved baseline journal PDF
│   │   ├── README.md                       <-- Paper build and metadata guide
│   │   └── figures/                        <-- High-resolution vector & PNG graphics
│   │       ├── architecture_cette.pdf
│   │       ├── fig_split_audit.pdf
│   │       ├── fig_capacity.pdf
│   │       ├── 04_reliability.pdf
│   │       ├── confusion_matrices_dual.pdf
│   │       ├── error_decorrelation.pdf
│   │       └── training_dynamics.pdf
│   └── study/                              <-- Research design playbooks, reviews, & analysis notes
│
├── datasets/
│   ├── raw/                                <-- Original datasets (CMTE-NITANP, CMTET, DravidianTamil)
│   ├── processed/                          <-- Normalized and cleaned datasets
│   └── splits/                             <-- Leak-free Split B and baseline Split A (JSONL)
│       ├── CMTE-NITANP/
│       └── CMTET/
│
├── Implementation/
│   ├── baseline/                           <-- Initial exploratory baselines (SVM, LogReg, MuRIL, CNN)
│   └── benchmark/                          <-- Core 21-model benchmark execution suite
│       ├── common/                         <-- Data loaders, metrics, evaluation runners
│       ├── audit/                          <-- Exact-Jaccard duplicate audit scripts
│       ├── classical/                      <-- 11 classical algorithms + feature ablations
│       ├── deep/                           <-- TextCNN, BiLSTM, CNN-BiLSTM implementations
│       ├── hf_transformers/                <-- Transformer fine-tuning & LoRA modules
│       ├── ensemble/                       <-- Late fusion, Dirichlet calibration, super-ensemble
│       ├── robustness/                     <-- Orthographic perturbation generation & test suite
│       ├── cross_corpus/                   <-- NITANP -> CMTET zero-shot domain transfer
│       ├── analysis/                       <-- Aggregation, statistical tests, report generator
│       └── run_queue.py                    <-- Automated multi-GPU execution queue
│
├── preprocessing/                          <-- Signal-preserving normalization scripts
└── report/                                 <-- Institutional technical reports & summaries
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ or 12.1+ recommended for transformer training.

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ShaikMohammad786/Tenglish_Codemixed_Sentiment_Analysis.git
cd Tenglish_Codemixed_Sentiment_Analysis

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r Implementation/baseline/requirements.txt
```

---

## 💻 Reproducing Experiments

### Step 1: Verify Zero-Leakage Split
```bash
cd Implementation/benchmark
python audit/verify_split_exact_jaccard.py
```

### Step 2: Run Classical Baselines & Linear Models (Fast, < 1 min)
```bash
python classical/ablation_features.py
python classical/linearsvc/run.py --phase final --dataset CMTE-NITANP
```

### Step 3: Run Transformer Backbones & CETTE Ensemble
```bash
# Execute benchmark queue across CPU and GPU lanes
python run_queue.py --lane all --hours 14

# Evaluate CETTE late fusion with temperature calibration
python ensemble/evaluate_cette.py --dataset CMTE-NITANP --regime dual
```

### Step 4: Robustness & Cross-Corpus Evaluation
```bash
# Run orthographic perturbation robustness test
python robustness/make_perturbations.py
python robustness/eval_robustness.py

# Run CMTET cross-corpus zero-shot transfer
python cross_corpus/eval_cross_corpus.py
```

### Step 5: Regenerate Figures & Analysis
```bash
# Generate all master figures in vector PDF and 400 DPI PNG
python generate_paper_plots.py
```

---

## 📄 Compiling the Conference Paper

To compile the 10-page camera-ready IEEE conference manuscript from source:
```bash
cd architecture/paper
pdflatex CETTE_Conference_Paper.tex
pdflatex CETTE_Conference_Paper.tex
```
The compiled paper will be written to `architecture/paper/CETTE_Conference_Paper.pdf`.

---

## 📖 Citation

If you use our code, benchmarks, or methodology in your research, please cite our paper:

```bibtex
@inproceedings{mohammad2026cette,
  author    = {Shaik Mohammad and Shaik Abdullah Tayyeb and Panditi Jayakrishna and Prakash Babu Yandrapati},
  title     = {{CETTE}: Complementary-Error Tri-Transformer Ensemble and Dual-Regime Benchmarking for Telugu--English Code-Mixed Sentiment Analysis},
  booktitle = {Proceedings of the IEEE Conference on Language and Information Technologies},
  year      = {2026},
  address   = {Amaravati, India},
  publisher = {IEEE}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
All datasets retain their original licenses as documented in [`datasets/raw/SOURCES.md`](datasets/raw/SOURCES.md).
