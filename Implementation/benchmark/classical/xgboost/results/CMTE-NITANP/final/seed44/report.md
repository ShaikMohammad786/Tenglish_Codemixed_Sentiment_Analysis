# xgboost (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}` | frozen hash: `da0a691809224d21cbd78e4be5a9b8f005d8ab0577a434b86344451b3e731d25` | val macro-F1: 0.7700

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7705 | 0.7700 | 0.7703 | 0.6564 | 0.0700 |
| test | 0.7768 | 0.7767 | 0.7768 | 0.6655 | 0.0766 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7261    0.8323    0.7756      1258
     Neutral     0.7935    0.7160    0.7528      1095
    Positive     0.8319    0.7740    0.8019      1106

    accuracy                         0.7768      3459
   macro avg     0.7838    0.7741    0.7767      3459
weighted avg     0.7813    0.7768    0.7768      3459
```
