# extra_trees (word11) — final

seed: 45 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.8042

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8043 | 0.8042 | 0.8042 | 0.7057 | 0.0527 |
| test | 0.8077 | 0.8079 | 0.8076 | 0.7110 | 0.0537 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7873    0.8148    0.8008      1258
     Neutral     0.8119    0.7726    0.7918      1095
    Positive     0.8278    0.8345    0.8312      1106

    accuracy                         0.8077      3459
   macro avg     0.8090    0.8073    0.8079      3459
weighted avg     0.8080    0.8077    0.8076      3459
```
