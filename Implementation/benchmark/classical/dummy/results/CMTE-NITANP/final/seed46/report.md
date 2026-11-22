# dummy (word11) — final

seed: 46 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{}` | frozen hash: `26e475e4797ddfe5ebfd8d41be14f6768474f922c01bd0dfdf342a163ecd9e43` | val macro-F1: 0.1779

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.3640 | 0.1779 | 0.1943 | 0.0000 | 0.6360 |
| test | 0.3637 | 0.1778 | 0.1940 | 0.0000 | 0.6363 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.3637    1.0000    0.5334      1258
     Neutral     0.0000    0.0000    0.0000      1095
    Positive     0.0000    0.0000    0.0000      1106

    accuracy                         0.3637      3459
   macro avg     0.1212    0.3333    0.1778      3459
weighted avg     0.1323    0.3637    0.1940      3459
```
