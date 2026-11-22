# Baseline: CNN + BiLSTM

text field: `text_clean` | tokenizer: `wordpiece:google/muril-base-cased` | splits: `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/datasets/splits/CMTE-NITANP` | device: cpu | epochs: 5 | emb: 100 | filters: 100 | hidden: 64 | max_len: 40 | seed: 42

| split | rows (train/val/test) | vocab | tokens/item | trunc % | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 27,670 / 3,459 / 3,459 | 197,258 | 17.42 | 8.50 | 0.7812 | 0.7785 | 0.7779 |
| B | 27,670 / 3,459 / 3,459 | 197,258 | 17.86 | 9.71 | 0.7880 | 0.7942 | 0.7945 |

**Leakage gap (A - B): macro-F1 -0.0166 | accuracy -0.0157**

Best epoch chosen on val macro-F1; test never used for tuning.

## Per-class report

### Split A

```
              precision    recall  f1-score   support

    Negative     0.7560    0.8251    0.7891      1258
     Neutral     0.7634    0.7543    0.7588      1095
    Positive     0.8257    0.7495    0.7858      1106

    accuracy                         0.7785      3459
   macro avg     0.7817    0.7763    0.7779      3459
weighted avg     0.7806    0.7785    0.7784      3459
```

### Split B

```
              precision    recall  f1-score   support

    Negative     0.7627    0.8251    0.7927      1258
     Neutral     0.7752    0.7653    0.7702      1095
    Positive     0.8564    0.7875    0.8205      1106

    accuracy                         0.7942      3459
   macro avg     0.7981    0.7926    0.7945      3459
weighted avg     0.7966    0.7942    0.7945      3459
```
