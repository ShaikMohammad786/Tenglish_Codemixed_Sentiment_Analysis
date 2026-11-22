# IEEE manuscript — *The Capacity Illusion*

**Files**
- `CMTE_NITANP_IEEE.tex` — main manuscript (IEEEtran, 10 pt, journal option; embedded bibliography, no BibTeX needed).
- `CMTE_NITANP_IEEE.pdf` — compiled manuscript: **13 pages, 8 tables, 9 figures**, 0 undefined references, 0 overfull boxes.
- `figures/` — all figures used by the manuscript, plus two unused benchmark plots (`02_per_class_f1.png`, `07_learning_curves.png`).

**Compile**
```bash
cd architecture/paper
pdflatex CMTE_NITANP_IEEE.tex
pdflatex CMTE_NITANP_IEEE.tex
```
or `latexmk -pdf CMTE_NITANP_IEEE.tex`. The current PDF was compiled with Tectonic 0.17 (installed only in the session scratchpad; TeX Live is not installed system-wide).

**Before submission**
- Replace the author block and the `\markboth` venue placeholder (currently "IEEE Transactions on Affective Computing (under review)").
- The manuscript is a two-column IEEE journal paper; it is **13 pages** with 8 tables and 9 figures.
- The paper deliberately reports only the **abstract-level** claim of Rayala & Seshadri (2026) (+5% accuracy / +6% F1 on CMTE-NITANP). Their absolute Tables 6–7 values were not accessible, so they are not quoted. The IEEE 11346359 83.4% figure is likewise **not** used as a CMTE-NITANP record because the accessible record does not confirm the dataset/split.

**Where every number comes from**
| Manuscript item | Source |
|---|---|
| 21-configuration leaderboard, CIs, ECE, cost | `Implementation/benchmark/results/seed_summary.csv`, `master_results.csv` |
| Post-hoc best single (LinearSVC union) | `Implementation/benchmark/ensemble/preds/linearsvc_union/preds.npz`; computed 2026-10-24 |
| Paired tests vs LogReg, Holm | `Implementation/benchmark/results/stats/pairwise_vs_logreg.csv` |
| Feature ablation | `Implementation/benchmark/results/feature_ablation.csv` + the calibrated union run |
| Split audit (Split A 345 pairs, Split B 0) | `Implementation/benchmark/audit/results/CMTE-NITANP/split_audit.md` |
| Split construction, cluster stats, symdiff | `datasets/splits/CMTE-NITANP/split_report.md` |
| Robustness | `Implementation/benchmark/robustness/results/CMTE-NITANP/robustness.md` |
| Cross-corpus transfer | `Implementation/benchmark/cross_corpus/results/CMTET/cross_corpus.md` |
| Sensitivity | `Implementation/benchmark/sensitivity/results/CMTE-NITANP/sensitivity.md` |
| Ensemble / hybrid experiments | `Implementation/benchmark/ensemble/` |
| RQ answers, report | `Implementation/benchmark/reports/REPORT.md` |

**Headline numbers (paper)**
- **Best single-view model: calibrated LinearSVC, word+char union — macro-F1 0.8363, accuracy 0.8364, ECE 0.0372, 379.1K coefficients, 4.6 s end-to-end.**
- **Post-hoc multi-view variant: clean + PLD-normalized word/char union — 0.8392 macro-F1, 0.8393 accuracy, 752.4K coefficients, 12.2 s end-to-end (paper Results §F).**
- Best transformer: MuRIL full-FT — macro-F1 **0.8336**, accuracy **0.8338**, 53.9 min.
- Best single-view vs best transformer: **+0.0027 macro-F1 / +0.0026 accuracy**; McNemar **p = 0.659**; 95% bootstrap CI **[−0.0073, +0.0121]** — statistically tied, linear nominally ahead.
- Multi-view vs best transformer: **+0.0056 macro-F1**; McNemar p = 0.307; 95% CI **[−0.0044, +0.0150]** — not significant.
- TF-IDF LogReg: **0.8293** (word unigrams); **0.8324** (word+char union).
- Full-FT vs deterministic LogReg: +0.0043 macro-F1; McNemar p = 0.4367; Holm p = 0.8158 (test-set interval, not cross-seed).
- Capacity vs macro-F1: Spearman ρ = **0.306**, p = **0.504**.
- Split A: **345** exact-Jaccard ≥ 0.8 train–test pairs, **203** test rows with a twin in train. Split B: **0** and **0**.
- Input statistics use `text_clean`: train 10.46 tokens / 8 median, 66.7 chars / 47 median; vocabulary 45,868; test OOV 11.12%.

**Experimental frontier (paper Results §F + supporting artifacts)**
- Heterogeneous late-fusion ensemble (3 linear + 2 MuRIL + classical soft-voting): **85.00% accuracy / 85.01 macro-F1** uniform; **85.02 / 85.03** temperature-calibrated. This is a **selection-sensitive** result: strict val-only “best per family” is **84.77**, and a val-fitted simplex blend is **84.33**. It is real on leak-free Split B but not a single-model claim; it is reported as a post-hoc probe in paper Results §F.
- Feature-level hybrid (MuRIL CLS + linear probabilities, trained on validation, PCA64): **83.61 / 83.65**; PCA128: **82.86 / 82.90**. It did **not** beat late fusion; supporting artifact only.
- MuRIL full-FT at `max_len=96` on `text_clean` (frozen embeddings): early stop at epoch 2; best val **0.8231**, test **82.75 / 82.71** — worse than the 83.36/83.38 starting checkpoint.
- MuRIL full-FT at `max_len=96` on `text_norm`: epoch 1 val **0.8179**; stopped.
- `max_len=128` OOM’d at batch 4 and 2; freezing embeddings fit, but 128 only rescues one row over 96.
- **90% is not credible without leakage.** Claude’s ceiling review (`architecture/CLAUDE_90_CEILING_REVIEW.md`) finds 357/3,459 test rows where ≥80% of 108 trained models disagree with gold; realistic label-noise ceiling **~86.0–86.8%**.

**Final audit performed**
- All 21 leaderboard rows and all 20 pairwise rows parsed from the `.tex` and compared cell-for-cell to the CSVs: **0 mismatches**.
- The post-hoc LinearSVC-union row is outside the frozen 21-configuration matrix and is labelled as such.
- Truncation table independently recomputed with the MuRIL tokenizer, counting the two special tokens: **15.21 / 7.60 / 3.47 / 0.17 / 0.14%** at max_len 32/48/64/96/128.
- Every float has a `\ref`; every `\cite` has a `\bibitem`; no unused bibliography entries.
- `qpdf --check`: no syntax or stream errors. `pdftotext`: 0 `??`.
