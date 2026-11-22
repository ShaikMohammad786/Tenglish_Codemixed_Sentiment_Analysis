# Baseline: CNN + BiLSTM

text field: `text_clean` | splits: `datasets/splits/CMTE-NITANP` | device: cpu | epochs: 5 | emb: 100 | filters: 100 | hidden: 64 | max_len: 40 | seed: 42

| split | rows (train/val/test) | vocab | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|---:|
| A | 27,670 / 3,459 / 3,459 | 17,202 | 0.7804 | 0.7872 | 0.7871 |
| B | 27,670 / 3,459 / 3,459 | 17,300 | 0.7884 | 0.7858 | 0.7863 |

**Leakage gap (A - B): macro-F1 +0.0008 | accuracy +0.0014**

Best epoch chosen on val macro-F1; test never used for tuning.

## Per-class report

### Split A

```
              precision    recall  f1-score   support

    Negative     0.7707    0.8148    0.7921      1258
     Neutral     0.7583    0.7881    0.7730      1095
    Positive     0.8426    0.7550    0.7964      1106

    accuracy                         0.7872      3459
   macro avg     0.7905    0.7860    0.7871      3459
weighted avg     0.7898    0.7872    0.7874      3459
```

### Split B

```
              precision    recall  f1-score   support

    Negative     0.7605    0.8076    0.7833      1258
     Neutral     0.7551    0.7772    0.7660      1095
    Positive     0.8544    0.7694    0.8097      1106

    accuracy                         0.7858      3459
   macro avg     0.7900    0.7847    0.7863      3459
weighted avg     0.7888    0.7858    0.7863      3459
```
