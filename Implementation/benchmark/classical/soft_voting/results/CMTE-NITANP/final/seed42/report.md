# soft_voting (word11) — final

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{}` | frozen hash: `26e475e4797ddfe5ebfd8d41be14f6768474f922c01bd0dfdf342a163ecd9e43` | val macro-F1: 0.8115

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8115 | 0.8115 | 0.8115 | 0.7167 | 0.0591 |
| test | 0.8225 | 0.8227 | 0.8225 | 0.7331 | 0.0669 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8021    0.8315    0.8165      1258
     Neutral     0.8216    0.7909    0.8060      1095
    Positive     0.8474    0.8436    0.8455      1106

    accuracy                         0.8225      3459
   macro avg     0.8237    0.8220    0.8227      3459
weighted avg     0.8228    0.8225    0.8225      3459
```
