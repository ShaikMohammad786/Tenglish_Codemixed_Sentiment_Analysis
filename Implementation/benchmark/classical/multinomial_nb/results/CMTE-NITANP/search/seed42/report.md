# multinomial_nb (word11) — search

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'alpha': 1.0}` | frozen hash: `None` | val macro-F1: 0.7951

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7968 | 0.7951 | 0.7955 | 0.6965 | 0.0734 |
| test | 0.8130 | 0.8121 | 0.8122 | 0.7198 | 0.0861 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7985    0.8283    0.8131      1258
     Neutral     0.8420    0.7397    0.7876      1095
    Positive     0.8054    0.8680    0.8355      1106

    accuracy                         0.8130      3459
   macro avg     0.8153    0.8120    0.8121      3459
weighted avg     0.8145    0.8130    0.8122      3459
```

## Search history (val)

| params | val macro-F1 |
|---|---:|
| `{'alpha': 0.1}` | 0.7863 |
| `{'alpha': 0.5}` | 0.7945 |
| `{'alpha': 1.0}` | 0.7951 |
