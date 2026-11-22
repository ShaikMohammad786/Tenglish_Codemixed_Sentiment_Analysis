# extra_trees (word11) — final

seed: 43 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7984

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7985 | 0.7984 | 0.7984 | 0.6970 | 0.0480 |
| test | 0.8051 | 0.8053 | 0.8050 | 0.7071 | 0.0552 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7828    0.8108    0.7966      1258
     Neutral     0.8133    0.7680    0.7900      1095
    Positive     0.8235    0.8354    0.8294      1106

    accuracy                         0.8051      3459
   macro avg     0.8066    0.8048    0.8053      3459
weighted avg     0.8055    0.8051    0.8050      3459
```
