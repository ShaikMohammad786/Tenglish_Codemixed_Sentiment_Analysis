# soft_voting (word11) — final

seed: 45 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{}` | frozen hash: `26e475e4797ddfe5ebfd8d41be14f6768474f922c01bd0dfdf342a163ecd9e43` | val macro-F1: 0.8123

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8124 | 0.8123 | 0.8123 | 0.7180 | 0.0599 |
| test | 0.8231 | 0.8233 | 0.8230 | 0.7340 | 0.0687 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8012    0.8299    0.8153      1258
     Neutral     0.8249    0.7918    0.8080      1095
    Positive     0.8471    0.8463    0.8467      1106

    accuracy                         0.8231      3459
   macro avg     0.8244    0.8227    0.8233      3459
weighted avg     0.8234    0.8231    0.8230      3459
```
