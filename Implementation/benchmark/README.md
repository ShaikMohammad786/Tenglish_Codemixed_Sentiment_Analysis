# Benchmark suite — Romanized Telugu–English sentiment (CMTE-NITANP)

Reproducible, leakage-audited benchmark of classical ML, deep learning, and
pretrained transformer models for 3-class Romanized Tenglish sentiment.

## Key protocol rules

- **Split B only** (`datasets/splits/CMTE-NITANP/splitB_*.jsonl`), audited for exact
  train–test duplicate overlap (`audit/verify_split_exact_jaccard.py`).
- **Val-only tuning**; test evaluated once per final model.
- **Seeds never change hyperparameters**: `--phase search` freezes the winner to
  `results/<dataset>/best_config.json`; `--phase final` reuses it for every seed.
- **10 epochs max** for trainable models, early stopping (patience 2), best-on-val restore.
- **Per-epoch checkpoints** (`checkpoint_last.pt`, `checkpoint_best.pt`), atomic writes,
  `--resume auto`; completed runs are skipped unless `--force`.

## Layout

```
Implementation/benchmark/
├── common/                    shared: paths, io/logging, data+cache, metrics, stats, ckpt, runner
├── audit/                     exact-Jaccard split audit
├── classical/<model>/         run.py + results/<dataset>/<phase>/seed<k>/
├── deep/<model>/              run.py + results/...
├── hf_transformers/<model>/   run.py + results/...  (LoRA; muril_fullft = full FT gate)
├── robustness/                perturbation generation + robustness eval
├── cross_corpus/              NITANP -> CMTET zero-shot evaluator
├── analysis/aggregate.py      master CSVs, stats, figures, REPORT.md, LaTeX
├── run_queue.py               concurrent CPU + GPU lanes, time guard
├── validate_results.py        artifact validation
└── finalize_after_queue.sh    wait -> validate -> robustness -> cross-corpus -> ablation -> report
```

## Run

```bash
cd Implementation/benchmark
../../.venv/bin/python audit/verify_split_exact_jaccard.py
../../.venv/bin/python run_queue.py --lane all --hours 14 --job-timeout 7200
# after the queue:
../../.venv/bin/python validate_results.py
../../.venv/bin/python robustness/make_perturbations.py
../../.venv/bin/python robustness/eval_robustness.py
../../.venv/bin/python cross_corpus/eval_cross_corpus.py
../../.venv/bin/python classical/ablation_features.py
../../.venv/bin/python analysis/aggregate.py --plots
```

Re-running any command is safe: runs skip if `run_status.json` says `done`, and
training resumes from the last epoch checkpoint.

## Outputs

- `results/master_results.csv`, `results/seed_summary.csv`, `results/feature_ablation.csv`
- `results/stats/pairwise_vs_logreg.csv`
- `reports/REPORT.md` (RQ1–RQ8), `reports/tables/*.tex`
- `reports/plots/01–07_*.png` and `reports/plots/per_model/*.png` (one per run)
- `robustness/results/CMTE-NITANP/robustness.{json,md}`
- `cross_corpus/results/CMTET/cross_corpus.{json,md}`
- Per run: `metrics.json`, `history.json`, `per_class.txt`, `preds.npz`, `run.log`, `run_status.json`

## Environment

See `environment.lock` (package pins) and `environment.txt` (Python/torch/GPU).
