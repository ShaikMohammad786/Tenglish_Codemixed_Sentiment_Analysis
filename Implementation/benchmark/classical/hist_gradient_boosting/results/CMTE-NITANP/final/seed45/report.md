# hist_gradient_boosting (word11) — final

seed: 45 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'learning_rate': 0.1, 'max_iter': 200}` | frozen hash: `c01d49bac162c1501077e68730a8be0a476f5937ff5872c465ee37a0596956fe` | val macro-F1: 0.7915

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7913 | 0.7915 | 0.7913 | 0.6861 | 0.0236 |
| test | 0.8025 | 0.8028 | 0.8025 | 0.7031 | 0.0230 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7813    0.8092    0.7950      1258
     Neutral     0.7941    0.7680    0.7809      1095
    Positive     0.8359    0.8291    0.8325      1106

    accuracy                         0.8025      3459
   macro avg     0.8038    0.8021    0.8028      3459
weighted avg     0.8028    0.8025    0.8025      3459
```
