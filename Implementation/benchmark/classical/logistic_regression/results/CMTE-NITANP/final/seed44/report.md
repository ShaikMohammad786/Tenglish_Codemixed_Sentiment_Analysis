# logistic_regression (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'C': 1.0}` | frozen hash: `efa40aaef091839cd118a7ab219c67ba72a5b48eefe2bab9e60363a86262f46c` | val macro-F1: 0.8083

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8089 | 0.8083 | 0.8087 | 0.7129 | 0.0535 |
| test | 0.8291 | 0.8293 | 0.8291 | 0.7431 | 0.0680 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8044    0.8434    0.8234      1258
     Neutral     0.8267    0.7927    0.8093      1095
    Positive     0.8615    0.8490    0.8552      1106

    accuracy                         0.8291      3459
   macro avg     0.8308    0.8284    0.8293      3459
weighted avg     0.8297    0.8291    0.8291      3459
```
