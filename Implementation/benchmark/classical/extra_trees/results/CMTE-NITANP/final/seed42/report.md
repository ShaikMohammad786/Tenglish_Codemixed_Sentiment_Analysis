# extra_trees (word11) — final

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7999

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8002 | 0.7999 | 0.8001 | 0.6996 | 0.0495 |
| test | 0.8072 | 0.8073 | 0.8071 | 0.7101 | 0.0538 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7874    0.8124    0.7997      1258
     Neutral     0.8111    0.7726    0.7914      1095
    Positive     0.8265    0.8354    0.8309      1106

    accuracy                         0.8072      3459
   macro avg     0.8083    0.8068    0.8073      3459
weighted avg     0.8074    0.8072    0.8071      3459
```
