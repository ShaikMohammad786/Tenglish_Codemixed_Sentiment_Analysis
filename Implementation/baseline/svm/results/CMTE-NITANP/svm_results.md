# Baseline: TF-IDF + LinearSVC

text field: `text_clean` | splits: `datasets/splits/CMTE-NITANP` | seed: 42

| split | rows (train/val/test) | vocab | C | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| A | 27,670 / 3,459 / 3,459 | 16,781 | 0.1 | 0.8162 | 0.8173 | 0.8174 |
| B | 27,670 / 3,459 / 3,459 | 16,892 | 0.1 | 0.8118 | 0.8245 | 0.8248 |

**Leakage gap (A - B): macro-F1 -0.0074 | accuracy -0.0072**

C selected on val only; test never used for tuning.

## Per-class report

### Split A

```
              precision    recall  f1-score   support

    Negative     0.7895    0.8466    0.8170      1258
     Neutral     0.8117    0.7954    0.8035      1095
    Positive     0.8592    0.8056    0.8315      1106

    accuracy                         0.8173      3459
   macro avg     0.8201    0.8159    0.8174      3459
weighted avg     0.8188    0.8173    0.8174      3459
```

### Split B

```
              precision    recall  f1-score   support

    Negative     0.7946    0.8426    0.8179      1258
     Neutral     0.8253    0.7854    0.8049      1095
    Positive     0.8606    0.8427    0.8515      1106

    accuracy                         0.8245      3459
   macro avg     0.8268    0.8236    0.8248      3459
weighted avg     0.8254    0.8245    0.8245      3459
```
