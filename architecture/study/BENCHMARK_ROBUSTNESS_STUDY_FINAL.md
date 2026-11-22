# FINAL Plan — Multi-Model Benchmark & Robustness Study for Romanized Telugu–English Sentiment
## Leak-free (audited) CMTE-NITANP Split B · classical ML · deep learning · transformers · soft voting · calibration · abstention · cost · statistics

**Date:** 2026-10-11
**Status:** Final, after 3-agent review (Codex confirmed; Claude's earlier written verdicts + this round).
**Hard constraint:** **total end-to-end compute ≤ 15 hours** (RTX 2050 4 GB + 12-core CPU).
**Target output:** reproducible benchmark/robustness paper (workshop or journal tier) + code, logs, plots, LaTeX tables.

---

## 0. What changed after agent review (read this first)

Three corrections materially change the study — all are incorporated below.

1. **The 0.8293 LogReg baseline is word-unigram TF-IDF, not word+char.**
   `Implementation/baseline/logistic-regression/train_logreg.py` uses `TfidfVectorizer(min_df=2, sublinear_tf=True)` with sklearn defaults (`analyzer="word"`, `ngram_range=(1,1)`). All prior docs saying "word+char" are **wrong**. We will (a) report this correctly and (b) *add* genuine word+char and char-only variants as new experiments.

2. **"0.00% leakage" is a MinHash-LSH grouping claim, not an exact-duplicate guarantee.**
   `preprocessing/dedup_split.py` unions LSH candidates at `threshold=0.8` and never verifies exact shingle Jaccard. Multiple near-threshold LSH matches can chain, and LSH candidates may fall below 0.8 exact Jaccard. **Before calling Split B "leak-free," we run an exact-Jaccard audit** over all LSH candidate pairs and report:
   - pairs with exact Jaccard < 0.8 (approximate false positives),
   - the max exact Jaccard between any train and test pair,
   - borderline clusters (e.g. exact Jaccard 0.7–0.8).
   If the audit finds real leakage, rebuild the split with an exact-Jaccard verification pass. Until then the wording is **"grouped under MinHash-LSH (θ=0.8), pending exact-Jaccard audit."**

3. **Statistical tests must match the resampling design.**
   - One fixed Split B + repeated seeds ⇒ **paired bootstrap 95% CIs + McNemar** on the held-out test predictions.
   - `Nadeau–Bengio` corrected resampled t-test only for **actual repeated grouped CV** (classical family).
   - **Do not** run Friedman/Nemenyi on one dataset with seed repetitions (Nemenyi assumes multiple datasets). Use it only if we add ≥2 domains (e.g. NITANP + CMTET + one more), otherwise omit.

Codex also **rejected full test-time augmentation (TTA)**: extra inference cost, invites an ensemble critique, and spelling transforms risk circular evaluation. We drop TTA.

---

## 1. Objective & positioning

**What we build:** a leakage-audited, multi-family benchmark + robustness/reliability audit of Romanized Telugu–English (Tenglish) 3-class sentiment on CMTE-NITANP, with hyperparameter sensitivity, calibration, abstention/risk–coverage, orthographic robustness, and cost–benefit analysis.

**Not a new architecture.** This is a benchmark / reproducibility / reality-check paper. Honest venue expectation: **DravidianLangTech / VarDial workshop** (primary) → extended to **ESWA / ACM TALLIP / IEEE Access / SNAM**.

**Defensible gap (verified):** `DravidianCodeMix` (LRE 2022) benchmarked ML+transformers on Tamil/Malayalam/Kannada, 5-class, random split; `Trailblazer` (DravidianLangTech 2026) compared TF-IDF-SVM vs XLM-R on Tamil political text. **No leakage-audited, hyperparameter-sensitivity + calibration + robustness + cost study exists for Telugu.**

**Anchor anomaly (to be re-measured under the corrected harness):** word-unigram LogReg 0.8293 > MuRIL+LoRA 0.8167 on Split B **[V]**.

---

## 2. Non-negotiables

1. Split B only for primary results; Split A only for the leakage sanity check.
2. **Exact-Jaccard audit of the split before any "leak-free" claim.**
3. Val-only tuning; test touched once per final model.
4. Group-aware inner CV (`group_id`, `StratifiedGroupKFold`) for hyperparameter search.
5. ≥3 seeds (transformers) / ≥5 seeds (classical, deep).
6. All numbers emitted to JSON/CSV by code; no hand-entered results.
7. No synthetic data in evaluation; perturbations only in the robustness section, clearly labelled.

---

## 3. Trimmed model matrix (fits ≤15 h)

### 3.1 Classical (CPU; ~1.0 h total)
| Model | Grid |
|---|---|
| Dummy (majority) | — |
| Multinomial NB | alpha {0.1,0.5,1} |
| Logistic Regression | C {0.1,1,10} |
| LinearSVC | C {0.01,0.1,1} |
| SVM-RBF | C {1,10}, gamma {scale,1e-3} |
| Random Forest | 300 trees, max_depth {None,64}, min_samples_leaf {1,3} |
| Extra Trees | same |
| HistGradientBoosting | lr {0.05,0.1}, max_iter {200,400} |
| kNN (cosine) | k {5,15,30} |
| XGBoost / LightGBM *(only if installed)* | shallow, 300 rounds |

**Feature spaces (the real lever):** word TF-IDF (1–1, the *existing* default), word (1–2), char_wb (2–5), and word+char union. `sublinear_tf=True`, `min_df` tuned on train.

### 3.2 Soft voting (cheap; adds credibility)
```python
VotingClassifier(
  estimators=[('lr', LogisticRegression(...)),          # has predict_proba
              ('rf', RandomForestClassifier(300, n_jobs=-1)),  # averages tree probabilities (soft)
              ('et', ExtraTreesClassifier(300, n_jobs=-1)),
              ('hgb', HistGradientBoostingClassifier(...))],
  voting='soft', n_jobs=-1)
```
- **RF is already soft** (`predict_proba` averages tree class probabilities); `voting='soft'` is the ensemble level.
- `LinearSVC` has no `predict_proba`; if included, wrap it in `CalibratedClassifierCV(LinearSVC, cv=3, method='sigmoid')`. Otherwise use `SGDClassifier(loss='modified_huber')`.
- **Leakage trap avoided:** vectorizer + calibrator fit on train only; val used only for config/weight selection; no calibration on test.
- Optional: soft-vote weights tuned on val (reported; never on test).

### 3.3 Deep learning (PyTorch; ~0.5 h)
TextCNN, BiLSTM, CNN+BiLSTM (existing). Word-level and MuRIL-WordPiece where relevant; **max 10 epochs**, early stopping (patience 2) on val macro-F1, **restore best epoch**; 5 seeds.

### 3.4 Transformers (LoRA, bf16; ~3.2 h finals + ~1.2 h search)
| Model | HF id | Adaptation |
|---|---|---|
| MuRIL-base | `google/muril-base-cased` | LoRA |
| XLM-RoBERTa-base | `xlm-roberta-base` | LoRA |
| IndicBERT v2 | `ai4bharat/IndicBERTv2-MLM-only` (open) | LoRA |
| TeluguBERT | `l3cube-pune/telugu-bert` (open, Telugu-specific) | LoRA |
| mBERT | `bert-base-multilingual-cased` | LoRA |
| DistilmBERT | `distilbert-base-multilingual-cased` | LoRA |
| **Full-FT gate** | MuRIL-base | full FT (batch 8×4) |

*Note:* `ai4bharat/indic-bert` and `neuralspace-reverie/indic-transformers-te` are gated (401) without an HF token; the open equivalents above are used instead.

LoRA r=16, alpha=32, dropout 0.1, targets `["query","value"]`. Grid: lr {2e-4,4e-4}; **`max_epochs=10` with early stopping (patience 2) and best-epoch restore**; max_len 64.

---

## 4. ≤15-hour compute budget (arithmetic)

**Measured anchor:** MuRIL+LoRA ≈ **2.75 min/epoch** on Split B (45 min for A+B × 8 epochs **[V]**). With `max_epochs=10`, early stopping (patience 2) and best-epoch restore, runs average ~7 epochs ≈ **20 min**; DistilmBERT ≈ **10 min**.

| Job | Runs | Min/run | Total |
|---|---:|---:|---:|
| Exact-Jaccard split audit | 1 | 12 | 12 min |
| **Timed pilot** (MuRIL, 2 epochs) | 1 | 6 | 6 min |
| Transformer search (alt config × 1 seed × 4 models) | 4 | 20 | 80 min |
| Transformer finals (MuRIL/XLM-R/IndicBERT/mBERT × 3 seeds) | 12 | 20 | 240 min |
| DistilmBERT finals | 3 | 10 | 30 min |
| Full-FT gate (MuRIL, max 10 epochs) | 1 | 50 | 50 min |
| Classical grid + group-aware 3-fold CV | ~60 fits | — | 60 min |
| Soft voting + calibration | 1 | — | 18 min |
| Deep (3 models × 5 seeds, max 10 epochs) | 15 | ~2 | 30 min |
| CMTET zero-shot + robustness + calibration eval | — | — | 42 min |
| Aggregation + plots + report | — | — | 18 min |
| **Total** | | | **≈ 9.8 h** |

**Guardrails:** the runner tracks elapsed time and **stops optional jobs at 14.0 h** (optional = search variants, extra seeds); every trainable model **checkpoints each epoch and resumes**; if the pilot is slower than 2.75 min/epoch, drop DistilmBERT seeds or the search lane first.

---

## 5. Code & logging architecture

```
Implementation/benchmark/
├── common/{data,io,metrics,stats,plots}.py
├── audit/verify_split_exact_jaccard.py
├── classical/run_classical.py
├── deep/run_deep.py
├── hf_transformers/run_transformer.py
├── robustness/{make_perturbations,eval_robustness}.py
├── analysis/aggregate.py
├── configs/*.yaml
├── results/
│   ├── runs/<model>/<config>/<seed>/{metrics.json,per_class.txt,history.json,preds.npz,run.log}
│   ├── master_results.csv
│   ├── hparam_search.csv
│   └── stats/*.csv
└── reports/{REPORT.md,tables/*.tex,plots/*.png}
```

**Run manifest:** `run_id, model, family, config, seed, split, n_train/val/test, train_time_s, peak_vram_mb, peak_ram_mb, val_macro_f1, test_acc, test_macro_f1, mcc, ece, brier, per_class{P,R,F1}, history[]`.

**Batch processing:** models run from a queue (`configs/queue.yaml`), sequentially, with `run_id` checkpoint/resume; the runner writes a status line per job and a global elapsed-time budget.

---

## 6. Logs, reports, graphs

**Per run:** `.log`, `metrics.json`, `history.json`, `preds.npz` (paired tests).
**Aggregate:** `master_results.csv`, `hparam_search.csv`, `stats/`, `REPORT.md`, LaTeX tables.

**Figures (10):**
1. Model-family macro-F1 bar chart with 95% bootstrap CI.
2. Pairwise McNemar/bootstrap difference matrix (not Nemenyi on one dataset).
3. Learning curves (val macro-F1 per epoch).
4. Confusion matrices (best per family).
5. Reliability diagrams + ECE.
6. **Risk–coverage (abstention) curves** — F1 vs coverage.
7. Hyperparameter sensitivity heatmaps (C × feature space; lr × epochs).
8. Orthographic robustness Δ bars (clean / vowel / voicing / combined).
9. **Cost–benefit scatter** (macro-F1 vs train time, params, inference ms/1k).
10. Per-class F1 heatmap across models.

---

## 7. Statistics (corrected)

- 3–5 seeds → mean ± std.
- **Paired bootstrap 95% CIs** (1,000 resamples) on test macro-F1 and on pairwise differences.
- **McNemar** on paired test predictions (best vs each).
- **Nadeau–Bengio corrected resampled t-test** only for repeated grouped CV (classical).
- **No Friedman/Nemenyi** unless ≥2 independent domains are added; if added, use NITANP + CMTET (+ optional Tamil).
- Holm correction for multiple comparisons.

---

## 8. Cheap, credible additions (no TTA)

- **Calibration:** ECE (10 bins), Brier, reliability diagrams; temperature scaling on val.
- **Selective prediction / risk–coverage:** accuracy vs coverage at abstention thresholds — **first for Dravidian CMSA**.
- **Length- and class-stratified error analysis** (bucketed macro-F1, per-class confusion).
- **Spelling-family-held-out evaluation** (group spelling variants; evaluate on unseen families).
- **Seed-averaged transformer inference** (checkpoint ensemble; cheap, reduces variance).
- **Data-efficiency curves** (train-size subsampling for the classical family + one transformer; power-law fit).
- **Inference latency + peak memory** (cost column).

Explicitly **not** added: TTA (circular/ensemble critique), fusion/consistency (preempted).

---

## 9. Research questions

- **RQ1** Under an audited leak-free split, how do classical / deep / transformer families rank?
- **RQ2** Does the word-unigram lexical advantage persist when word+char features and calibration are added, and under LoRA vs full fine-tuning?
- **RQ3** Is accuracy predicted by capacity, or is the task lexical/orthographic?
- **RQ4** Which models are most robust to orthographic variation?
- **RQ5** What is the cost–benefit frontier (F1 per GPU-minute / per parameter)?
- **RQ6** Are models calibrated, and does calibration change the ranking?
- **RQ7** How much NITANP→CMTET transfer survives domain shift and label noise?
- **RQ8** What coverage/accuracy trade-off does abstention buy, and which family wins at 80/90% coverage?

---

## 10. Execution order (step-by-step)

1. **Split audit** — exact Jaccard over LSH candidate pairs; record max train–test Jaccard; rebuild if needed.
2. **Scaffold** `Implementation/benchmark/common/` + queue runner.
3. **Reproduce** LogReg / LinearSVC / CNN+BiLSTM / MuRIL under the new harness; **correct the word-unigram wording**.
4. **Timed pilot** — 1 MuRIL epoch to calibrate the real per-run time; adjust the budget if needed.
5. **Classical sweep + feature-space ablation + soft voting.**
6. **Deep sweep** × 5 seeds.
7. **Transformer search → finals** (3 seeds) under the time guard.
8. **Full-FT gate** (MuRIL).
9. **Robustness + calibration + abstention + CMTET.**
10. **Aggregate** → stats → 10 figures → LaTeX → `REPORT.md`; verify every number against `master_results.csv`.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Split not exactly leak-free | Exact-Jaccard audit first; rebuild if needed; reword until verified |
| Wrong baseline description | Correct to word-unigram; add word+char experiments |
| >10 h | Timed pilot; time guard at 9 h; 2 seeds for weakest models first |
| GPU OOM | LoRA only; full FT only MuRIL (batch 8×4, grad ckpt) |
| Seed noise | 3–5 seeds + CIs + paired tests |
| IndicBERT id/version | Pin HF revision; record in manifest |
| Thin novelty vs DravidianCodeMix-2022 | Differentiate: Telugu + audited split + calibration + abstention + cost; frame as reality check |
| Marginal/mixed result | Report honestly — the ranking *is* the finding |

---

## 12. Definition of done
- [ ] Exact-Jaccard split audit reported.
- [ ] All models on Split B with 3–5 seeds; everything in `master_results.csv`.
- [ ] Soft-voting ensemble + calibration + abstention results.
- [ ] Corrected statistics (bootstrap, McNemar; Nadeau–Bengio only for repeated CV).
- [ ] Robustness, cost, CMTET, and 10 figures/LaTeX tables generated.
- [ ] `REPORT.md` verified against artifacts; code/configs/seeds committed.

---

## 13. Implementation requirements (harness) — non-negotiable

### 13.1 Epochs & best-epoch selection
- **`MAX_EPOCHS = 10` for every trainable model** (TextCNN/BiLSTM/CNN+BiLSTM, all transformers, full-FT gate).
- Early stopping: `patience = 2`, `min_epochs = 3`, metric = **val macro-F1**.
- Track `best_epoch` / `best_val_macro_f1`; **restore best weights before touching test**.
- Save full per-epoch history (`history.json`) and an event stream (`events.jsonl`).

### 13.2 Caching (never recompute the same thing)
- **Tokenization cache:** key = `SHA256(tokenizer_revision, max_len, text_field, split, dataset)`; store `input_ids`, `attention_mask`, `labels` as `.pt` in `cache/tokenized/<key>/`.
- **Feature cache:** sparse TF-IDF matrices (`.npz`) + fitted vectorizer (joblib) in `cache/features/<key>/`.
- **Fitted-model cache:** classical pipelines (joblib) in `cache/models/<key>/`.
- **Prediction cache:** `preds.npz` per run (reused for paired tests).
- `--force` invalidates; otherwise reuse when key + checksum match. HF weights use the standard HF cache with pinned revisions.

### 13.3 Checkpointing & resume
- Every epoch writes **atomically** (`.tmp` → `os.replace`):
  - `checkpoint_last.pt` — LoRA/model state, head, optimizer, scheduler, scaler, epoch, `best_val`, `best_epoch`, RNG states.
  - `checkpoint_best.pt` — best-on-val state.
- `--resume auto` (default): resume if `checkpoint_last.pt` exists and config hash matches; else fresh. `--resume never|path`.
- `run_status.json` = `{pending|running|done|failed, started, ended, elapsed}`; the runner **skips done runs** unless `--redo`.
- Interrupted sessions cost at most one epoch.

### 13.4 Parallel CPU + GPU lanes
- Two lanes run **concurrently**:
  - `lane=cpu`: classical sweep, feature prep, deep-on-CPU, plots.
  - `lane=gpu`: transformers + full-FT gate.
- GPU process: `CUDA_VISIBLE_DEVICES=0`, `torch.set_num_threads(2)`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`.
- CPU lane: `n_jobs = max(1, cpu_count - 2)` (sklearn), `OMP/MKL = cpu_count - 2`.
- `results/.gpu.lock` prevents two GPU jobs (stale-lock detection by PID).
- `run_queue.py` spawns both lanes and writes `logs/lane_cpu.log`, `logs/lane_gpu.log`, merged `logs/master.log`.

### 13.5 Progress & logging (visible in cmd)
- Every run: `tee` stdout+stderr into `run.log` (timestamped) + structured `events.jsonl`.
- tqdm bars per epoch/batch (auto-off when non-TTY; `--quiet`).
- **Status board after each epoch:** `[model/config/seed] epoch k/10 | val_f1=… | best=… | elapsed=… | ETA=… | VRAM=…MB`.
- Global elapsed-time guard logged at each job boundary; optional jobs stop at 14.0 h.
- Failures: full traceback to `run.log`, `run_status=failed`, non-zero exit; the queue continues (unless `--fail-fast`).

### 13.6 Reports & graphs (auto-generated from artifacts)
- Per run: `metrics.json`, `per_class.txt`, `history.json`, `preds.npz`, `run.log`.
- Master: `master_results.csv`, `hparam_search.csv`, `stats/*.csv`.
- `reports/REPORT.md` generated from the artifacts — tables for per-model, per-family, best, calibration, robustness, cost, hyperparameter sensitivity, CMTET.
- LaTeX tables in `reports/tables/*.tex`; 10 figures in `reports/plots/*.png` (as §6).
- `reports/INDEX.md` lists every artifact with checksums.

### 13.7 Reproducibility
- Seeds 42–46 (classical/deep) and 42–44 (transformers); seed Python/NumPy/Torch; deterministic flags where feasible.
- Pin HF revisions + package versions (`environment.lock`); record git commit + config hash in every manifest.
- A run is "done" only if `metrics.json`, `history.json`, and `preds.npz` all exist and validate.

### 13.9 Seed vs hyperparameter policy (critical for publishability)
- **Seeds never change hyperparameters.** Search/tune ONCE per model (seed 42) on **validation only**, then **freeze** the winner to `results/<dataset>/best_config.json` (params + feature + provenance hash).
- **Final runs** (seeds 42–46 classical/deep, 42–44 transformers) load the frozen config; only parameter init, data shuffling, and dropout vary.
- Every run records `phase`, `params`/`hyper`, and `frozen_hash`; the aggregator groups only `phase=final` runs for mean ± std.
- Result: seed variance is reported separately from tuning variance (the defensible benchmark protocol).

### 13.10 Artifact validation
- `validate_results.py` checks every run's schema, monotone history, best-epoch restore, and that test metrics match `preds.npz`.
- Any mismatch fails the aggregate step (no report is generated from invalid artifacts).
