# random_forest (word11) — final

seed: 46 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7876

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7881 | 0.7876 | 0.7879 | 0.6815 | 0.0339 |
| test | 0.7942 | 0.7940 | 0.7940 | 0.6906 | 0.0457 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7673    0.8203    0.7929      1258
     Neutral     0.7994    0.7498    0.7738      1095
    Positive     0.8224    0.8083    0.8153      1106

    accuracy                         0.7942      3459
   macro avg     0.7964    0.7928    0.7940      3459
weighted avg     0.7951    0.7942    0.7940      3459
```
