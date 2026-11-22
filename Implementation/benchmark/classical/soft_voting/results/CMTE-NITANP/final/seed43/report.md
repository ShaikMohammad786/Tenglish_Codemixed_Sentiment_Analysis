# soft_voting (word11) — final

seed: 43 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{}` | frozen hash: `26e475e4797ddfe5ebfd8d41be14f6768474f922c01bd0dfdf342a163ecd9e43` | val macro-F1: 0.8129

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8130 | 0.8129 | 0.8129 | 0.7189 | 0.0591 |
| test | 0.8236 | 0.8239 | 0.8236 | 0.7349 | 0.0671 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8011    0.8291    0.8148      1258
     Neutral     0.8262    0.7900    0.8077      1095
    Positive     0.8477    0.8508    0.8493      1106

    accuracy                         0.8236      3459
   macro avg     0.8250    0.8233    0.8239      3459
weighted avg     0.8239    0.8236    0.8236      3459
```
