# soft_voting (word11) — final

seed: 46 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{}` | frozen hash: `26e475e4797ddfe5ebfd8d41be14f6768474f922c01bd0dfdf342a163ecd9e43` | val macro-F1: 0.8132

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8132 | 0.8132 | 0.8132 | 0.7194 | 0.0613 |
| test | 0.8219 | 0.8221 | 0.8219 | 0.7323 | 0.0673 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8006    0.8299    0.8150      1258
     Neutral     0.8219    0.7881    0.8047      1095
    Positive     0.8471    0.8463    0.8467      1106

    accuracy                         0.8219      3459
   macro avg     0.8232    0.8214    0.8221      3459
weighted avg     0.8222    0.8219    0.8219      3459
```
