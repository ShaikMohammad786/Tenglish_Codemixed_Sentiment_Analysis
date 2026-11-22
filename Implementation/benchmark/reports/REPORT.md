# Benchmark report — Romanized Telugu–English sentiment (CMTE-NITANP Split B)

Auto-generated from `Implementation/benchmark/**/results/**/metrics.json`.

## Research-question answers (auto-derived)

- **RQ1 (ranking):** best = `transformers/muril_fullft` at 0.8336 ± 0.0000.
- **RQ2 (lexical vs transformer):** best classical `logistic_regression` 0.8293 vs best transformer `muril_fullft` 0.8336 (Δ=-0.0043).
- **Significance:** 16/20 models differ from LogReg at p<0.05 (paired McNemar on the test set).
- **RQ3 (capacity):** Spearman ρ=0.306 (p=0.504) between parameter count and macro-F1 → capacity does not predict accuracy.
- **RQ6 (calibration):** best-calibrated `classical/lightgbm` ECE=0.0132.
- **RQ5 (cost–benefit):** best F1-per-second `multinomial_nb` (0.8121 in 0s).
- **RQ4 (robustness):** smallest worst-case degradation `transformers/distilmbert_lora` (max Δ=+0.1599 across variants).
- **RQ7 (CMTET transfer):** best zero-shot `transformers/indicbert_lora` F1=0.6380 (drop +0.1754).
- **RQ8 (abstention):** see `05_risk_coverage.png` and the per-run `risk_coverage` blocks.

## Seed summary (final runs, mean ± std, 95% bootstrap CI)

| family | model | seeds | macro-F1 | 95% CI | accuracy | ECE |
|---|---|---:|---:|---|---:|---:|
| transformers | muril_fullft | 1 | 0.8336 ± 0.0000 | [0.8211, 0.8467] | 0.8338 | 0.0702 |
| classical | logistic_regression | 5 | 0.8293 ± 0.0000 | [0.8182, 0.8419] | 0.8291 | 0.0680 |
| classical | linear_svc | 5 | 0.8231 ± 0.0000 | [0.8117, 0.8366] | 0.8231 | 0.0431 |
| classical | soft_voting | 5 | 0.8231 ± 0.0007 | [0.8112, 0.8348] | 0.8229 | 0.0677 |
| transformers | telugu_bert_lora | 3 | 0.8196 ± 0.0043 | [0.8026, 0.8287] | 0.8193 | 0.0634 |
| transformers | muril_lora | 3 | 0.8165 ± 0.0080 | [0.8114, 0.8354] | 0.8167 | 0.0540 |
| transformers | indicbert_lora | 3 | 0.8131 ± 0.0025 | [0.8010, 0.8264] | 0.8131 | 0.0589 |
| classical | multinomial_nb | 5 | 0.8121 ± 0.0000 | [0.7992, 0.8253] | 0.8130 | 0.0861 |
| transformers | xlmr_lora | 3 | 0.8103 ± 0.0066 | [0.7900, 0.8167] | 0.8102 | 0.0547 |
| transformers | mbert_lora | 3 | 0.8098 ± 0.0044 | [0.7935, 0.8208] | 0.8099 | 0.0657 |
| transformers | distilmbert_lora | 3 | 0.8092 ± 0.0022 | [0.7946, 0.8196] | 0.8093 | 0.0541 |
| classical | extra_trees | 5 | 0.8074 ± 0.0017 | [0.7952, 0.8217] | 0.8072 | 0.0548 |
| classical | lightgbm | 5 | 0.8048 ± 0.0000 | [0.7928, 0.8174] | 0.8049 | 0.0132 |
| classical | hist_gradient_boosting | 5 | 0.8018 ± 0.0013 | [0.7897, 0.8157] | 0.8016 | 0.0220 |
| deep | bilstm | 5 | 0.7983 ± 0.0042 | [0.7897, 0.8163] | 0.7983 | 0.0603 |
| classical | random_forest | 5 | 0.7968 ± 0.0022 | [0.7826, 0.8084] | 0.7970 | 0.0429 |
| deep | textcnn | 5 | 0.7928 ± 0.0036 | [0.7757, 0.8023] | 0.7929 | 0.0872 |
| deep | cnn_bilstm | 5 | 0.7846 ± 0.0040 | [0.7700, 0.7960] | 0.7846 | 0.0641 |
| classical | xgboost | 5 | 0.7770 ± 0.0024 | [0.7640, 0.7910] | 0.7771 | 0.0759 |
| classical | knn | 1 | 0.7755 ± 0.0000 | [0.7622, 0.7897] | 0.7759 | 0.0665 |
| classical | dummy | 5 | 0.1778 ± 0.0000 | [0.1722, 0.1835] | 0.3637 | 0.6363 |

## Paired tests vs LogReg (seed 42; paired bootstrap + McNemar)

| model | Δ macro-F1 | 95% CI | McNemar p | Holm p |
|---|---:|---|---:|---:|
| classical/dummy | -0.6518 | [-0.6645, -0.6386] | 0.0000 | 0.0000 |
| classical/extra_trees | -0.0223 | [-0.0318, -0.0128] | 0.0000 | 0.0000 |
| classical/hist_gradient_boosting | -0.0274 | [-0.0383, -0.0186] | 0.0000 | 0.0000 |
| classical/knn | -0.0541 | [-0.0652, -0.0424] | 0.0000 | 0.0000 |
| classical/lightgbm | -0.0249 | [-0.0341, -0.0156] | 0.0000 | 0.0000 |
| classical/linear_svc | -0.0061 | [-0.0120, -0.0005] | 0.0622 | 0.2488 |
| classical/multinomial_nb | -0.0174 | [-0.0265, -0.0074] | 0.0023 | 0.0161 |
| classical/random_forest | -0.0344 | [-0.0441, -0.0238] | 0.0000 | 0.0000 |
| classical/soft_voting | -0.0069 | [-0.0153, +0.0013] | 0.1495 | 0.4485 |
| classical/xgboost | -0.0523 | [-0.0640, -0.0416] | 0.0000 | 0.0000 |
| deep/bilstm | -0.0273 | [-0.0386, -0.0170] | 0.0000 | 0.0000 |
| deep/cnn_bilstm | -0.0466 | [-0.0576, -0.0356] | 0.0000 | 0.0000 |
| deep/textcnn | -0.0402 | [-0.0511, -0.0289] | 0.0000 | 0.0000 |
| transformers/distilmbert_lora | -0.0221 | [-0.0332, -0.0104] | 0.0006 | 0.0048 |
| transformers/indicbert_lora | -0.0164 | [-0.0292, -0.0042] | 0.0141 | 0.0846 |
| transformers/mbert_lora | -0.0235 | [-0.0356, -0.0102] | 0.0003 | 0.0027 |
| transformers/muril_fullft | +0.0039 | [-0.0062, +0.0140] | 0.4367 | 0.8158 |
| transformers/muril_lora | -0.0058 | [-0.0170, +0.0047] | 0.4079 | 0.8158 |
| transformers/telugu_bert_lora | -0.0147 | [-0.0263, -0.0034] | 0.0203 | 0.1015 |
| transformers/xlmr_lora | -0.0264 | [-0.0389, -0.0148] | 0.0000 | 0.0000 |
## Hyperparameter sensitivity (val only, fixed epoch budget)

| model | config | lr | max_len | batch | eff. batch | best epoch | val macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| muril_lora | baseline_main | 0.0004 | 64 | 16 | 32 | 8 | 0.8171 |
| muril_lora | lr2e4_len64_bs16 | 0.0002 | 64 | 16 | 32 | 5 | 0.7693 |
| muril_lora | lr4e4_len96_bs16 | 0.0004 | 96 | 16 | 32 | 5 | 0.7805 |
| muril_lora | lr4e4_len64_bs8 | 0.0004 | 64 | 8 | 32 | 5 | 0.7986 |
| mbert_lora | baseline_main | 0.0004 | 64 | 16 | 32 | 3 | 0.8061 |
| mbert_lora | lr2e4_len64_bs16 | 0.0002 | 64 | 16 | 32 | 5 | 0.7976 |
| mbert_lora | lr4e4_len96_bs16 | 0.0004 | 96 | 16 | 32 | 5 | 0.8022 |
| mbert_lora | lr4e4_len64_bs8 | 0.0004 | 64 | 8 | 32 | 5 | 0.7971 |
| distilmbert_lora | baseline_main | 0.0004 | 64 | 16 | 32 | 6 | 0.8109 |
| distilmbert_lora | lr2e4_len64_bs16 | 0.0002 | 64 | 16 | 32 | 5 | 0.8008 |
| distilmbert_lora | lr4e4_len96_bs16 | 0.0004 | 96 | 16 | 32 | 5 | 0.8092 |
| distilmbert_lora | lr4e4_len64_bs8 | 0.0004 | 64 | 8 | 32 | 5 | 0.8097 |


## External reference points (published; NOT reproduced here)

| source | setting | reported |
|---|---|---|
| Rayala & Seshadri (Springer 2025) | Telugu–English transformers on CMTE-NITANP | +5% acc / +6% F1 vs baselines (as reported) |
| IEEE 11346359 | Telugu–English hybrid/ensemble | 83.4% accuracy (best ensemble) |
| DravidianCodeMix (LRE 2022) | Tamil/Malayalam/Kannada, 5-class, random split | multi-model benchmark (not Telugu) |
| Rahul & Adebayo (DravidianLangTech 2026) | Hinglish robustness | XLM-R 99.7% in-dist → 42.5% adversarial |

These are context only; **our numbers use a different, leakage-audited protocol** and are not directly comparable.


## Figures

- `01_model_comparison.png`
- `02_per_class_f1.png`
- `03_confusions.png`
- `04_reliability.png`
- `05_risk_coverage.png`
- `06_cost_benefit.png`
- `07_learning_curves.png`
- `08_sensitivity.png`
