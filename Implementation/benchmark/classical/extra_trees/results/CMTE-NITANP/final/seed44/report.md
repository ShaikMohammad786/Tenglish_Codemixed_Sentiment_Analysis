# extra_trees (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7998

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8002 | 0.7998 | 0.8001 | 0.6996 | 0.0496 |
| test | 0.8098 | 0.8099 | 0.8096 | 0.7140 | 0.0559 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7903    0.8148    0.8023      1258
     Neutral     0.8156    0.7753    0.7949      1095
    Positive     0.8269    0.8382    0.8325      1106

    accuracy                         0.8098      3459
   macro avg     0.8109    0.8094    0.8099      3459
weighted avg     0.8100    0.8098    0.8096      3459
```
