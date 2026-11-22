# Baseline: MuRIL-base + LoRA

model: `google/muril-base-cased` | adapter: LoRA r=16 alpha=32 (query, value) | text field: `text_clean` | splits: `/mnt/M/MOHAMMAD/OmarchyWinProjects/Urop/datasets/splits/CMTE-NITANP` | device: cuda (bfloat16) | epochs: 8 | max_len: 64 | batch: 16x2 | lr: 0.0004 | seed: 42

trainable params: 592,131 / 238,150,662 (0.25%)

| split | rows (train/val/test) | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---:|---:|---:|
| A | 27,670 / 3,459 / 3,459 | 0.8182 | 0.8150 | 0.8151 |
| B | 27,670 / 3,459 / 3,459 | 0.8098 | 0.8167 | 0.8167 |

**Leakage gap (A - B): macro-F1 -0.0016 | accuracy -0.0017**

Best checkpoint chosen on val macro-F1; test never used for tuning.

## Per-class report

### Split A

```
              precision    recall  f1-score   support

    Negative     0.8185    0.8068    0.8127      1258
     Neutral     0.7886    0.8073    0.7978      1095
    Positive     0.8379    0.8318    0.8348      1106

    accuracy                         0.8150      3459
   macro avg     0.8150    0.8153    0.8151      3459
weighted avg     0.8152    0.8150    0.8151      3459
```

### Split B

```
              precision    recall  f1-score   support

    Negative     0.8195    0.8084    0.8139      1258
     Neutral     0.7879    0.7936    0.7907      1095
    Positive     0.8422    0.8490    0.8456      1106

    accuracy                         0.8167      3459
   macro avg     0.8165    0.8170    0.8167      3459
weighted avg     0.8167    0.8167    0.8167      3459
```
