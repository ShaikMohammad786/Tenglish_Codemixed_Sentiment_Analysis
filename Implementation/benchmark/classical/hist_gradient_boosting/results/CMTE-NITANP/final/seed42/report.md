# hist_gradient_boosting (word11) — final

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'learning_rate': 0.1, 'max_iter': 200}` | frozen hash: `c01d49bac162c1501077e68730a8be0a476f5937ff5872c465ee37a0596956fe` | val macro-F1: 0.7913

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7913 | 0.7913 | 0.7913 | 0.6861 | 0.0240 |
| test | 0.8020 | 0.8022 | 0.8019 | 0.7022 | 0.0199 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7782    0.8116    0.7946      1258
     Neutral     0.7973    0.7653    0.7810      1095
    Positive     0.8349    0.8273    0.8311      1106

    accuracy                         0.8020      3459
   macro avg     0.8035    0.8014    0.8022      3459
weighted avg     0.8024    0.8020    0.8019      3459
```
