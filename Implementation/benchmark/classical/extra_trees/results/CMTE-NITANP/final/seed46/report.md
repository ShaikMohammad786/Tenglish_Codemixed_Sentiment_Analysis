# extra_trees (word11) — final

seed: 46 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.8006

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.8008 | 0.8006 | 0.8007 | 0.7005 | 0.0515 |
| test | 0.8063 | 0.8065 | 0.8062 | 0.7088 | 0.0556 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7859    0.8140    0.7997      1258
     Neutral     0.8078    0.7717    0.7894      1095
    Positive     0.8288    0.8318    0.8303      1106

    accuracy                         0.8063      3459
   macro avg     0.8075    0.8058    0.8065      3459
weighted avg     0.8066    0.8063    0.8062      3459
```
