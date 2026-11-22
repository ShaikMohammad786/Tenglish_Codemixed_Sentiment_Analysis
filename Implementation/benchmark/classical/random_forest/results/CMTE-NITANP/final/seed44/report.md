# random_forest (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7881

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7887 | 0.7881 | 0.7885 | 0.6824 | 0.0327 |
| test | 0.7979 | 0.7977 | 0.7977 | 0.6963 | 0.0428 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7726    0.8235    0.7972      1258
     Neutral     0.8041    0.7498    0.7760      1095
    Positive     0.8232    0.8165    0.8198      1106

    accuracy                         0.7979      3459
   macro avg     0.7999    0.7966    0.7977      3459
weighted avg     0.7987    0.7979    0.7977      3459
```
