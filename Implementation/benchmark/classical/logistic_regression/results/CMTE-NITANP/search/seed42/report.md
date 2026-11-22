# logistic_regression (word11) — search

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'C': 1.0}` | frozen hash: `None` | val macro-F1: 0.8083

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

## Search history (val)

| params | val macro-F1 |
|---|---:|
| `{'C': 0.1}` | 0.7811 |
| `{'C': 1.0}` | 0.8083 |
| `{'C': 10.0}` | 0.7926 |
