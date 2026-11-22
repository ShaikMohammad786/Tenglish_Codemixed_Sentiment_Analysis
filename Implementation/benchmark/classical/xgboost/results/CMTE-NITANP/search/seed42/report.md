# xgboost (word11) — search

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}` | frozen hash: `None` | val macro-F1: 0.7709

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7713 | 0.7709 | 0.7712 | 0.6578 | 0.0708 |
| test | 0.7774 | 0.7773 | 0.7773 | 0.6662 | 0.0752 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7280    0.8299    0.7756      1258
     Neutral     0.7927    0.7160    0.7524      1095
    Positive     0.8311    0.7785    0.8039      1106

    accuracy                         0.7774      3459
   macro avg     0.7839    0.7748    0.7773      3459
weighted avg     0.7815    0.7774    0.7773      3459
```

## Search history (val)

| params | val macro-F1 |
|---|---:|
| `{'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}` | 0.7709 |
| `{'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300}` | 0.7080 |
