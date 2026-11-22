# random_forest (word11) — final

seed: 45 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7914

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7918 | 0.7914 | 0.7917 | 0.6871 | 0.0356 |
| test | 0.7999 | 0.7997 | 0.7997 | 0.6993 | 0.0439 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7733    0.8243    0.7980      1258
     Neutral     0.8041    0.7498    0.7760      1095
    Positive     0.8286    0.8219    0.8252      1106

    accuracy                         0.7999      3459
   macro avg     0.8020    0.7987    0.7997      3459
weighted avg     0.8007    0.7999    0.7997      3459
```
