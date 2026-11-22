# hist_gradient_boosting (word11) — final

seed: 43 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'learning_rate': 0.1, 'max_iter': 200}` | frozen hash: `c01d49bac162c1501077e68730a8be0a476f5937ff5872c465ee37a0596956fe` | val macro-F1: 0.7953

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7953 | 0.7953 | 0.7953 | 0.6923 | 0.0296 |
| test | 0.8020 | 0.8023 | 0.8019 | 0.7022 | 0.0227 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7800    0.8060    0.7928      1258
     Neutral     0.7970    0.7671    0.7818      1095
    Positive     0.8326    0.8318    0.8322      1106

    accuracy                         0.8020      3459
   macro avg     0.8032    0.8017    0.8023      3459
weighted avg     0.8022    0.8020    0.8019      3459
```
