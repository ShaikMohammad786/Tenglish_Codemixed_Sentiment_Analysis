# Baseline: TF-IDF + Logistic Regression

text field: `text_clean` | splits: `datasets/splits/CMTE-NITANP` | seed: 42

| split | rows (train/val/test) | vocab | C | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| A | 27,670 / 3,459 / 3,459 | 16,781 | 1.0 | 0.8170 | 0.8202 | 0.8204 |
| B | 27,670 / 3,459 / 3,459 | 16,892 | 1.0 | 0.8083 | 0.8291 | 0.8293 |

**Leakage gap (A - B): macro-F1 -0.0089 | accuracy -0.0089**

C selected on val only; test never used for tuning.

## Per-class report

### Split A

```
              precision    recall  f1-score   support

    Negative     0.7937    0.8474    0.8197      1258
     Neutral     0.8031    0.8009    0.8020      1095
    Positive     0.8730    0.8083    0.8394      1106

    accuracy                         0.8202      3459
   macro avg     0.8233    0.8189    0.8204      3459
weighted avg     0.8221    0.8202    0.8204      3459
```

### Split B

```
              precision    recall  f1-score   support

    Negative     0.8044    0.8434    0.8234      1258
     Neutral     0.8267    0.7927    0.8093      1095
    Positive     0.8615    0.8490    0.8552      1106

    accuracy                         0.8291      3459
   macro avg     0.8308    0.8284    0.8293      3459
weighted avg     0.8297    0.8291    0.8291      3459
```
