# random_forest (word11) — final

seed: 43 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `635d3511e813b5f99d44cfab3ecde5dfdb4d7ea8f008c195f9ec423655c9cebc` | val macro-F1: 0.7869

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7872 | 0.7869 | 0.7872 | 0.6801 | 0.0304 |
| test | 0.7973 | 0.7972 | 0.7972 | 0.6954 | 0.0405 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7721    0.8188    0.7948      1258
     Neutral     0.8016    0.7525    0.7763      1095
    Positive     0.8241    0.8174    0.8207      1106

    accuracy                         0.7973      3459
   macro avg     0.7992    0.7962    0.7972      3459
weighted avg     0.7980    0.7973    0.7972      3459
```
