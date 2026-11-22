# soft_voting (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{}` | frozen hash: `26e475e4797ddfe5ebfd8d41be14f6768474f922c01bd0dfdf342a163ecd9e43` | val macro-F1: 0.8123

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8124 | 0.8123 | 0.8123 | 0.7180 | 0.0602 |
| test | 0.8234 | 0.8235 | 0.8233 | 0.7344 | 0.0686 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8040    0.8315    0.8175      1258
     Neutral     0.8221    0.7890    0.8052      1095
    Positive     0.8473    0.8481    0.8477      1106

    accuracy                         0.8234      3459
   macro avg     0.8245    0.8229    0.8235      3459
weighted avg     0.8236    0.8234    0.8233      3459
```
