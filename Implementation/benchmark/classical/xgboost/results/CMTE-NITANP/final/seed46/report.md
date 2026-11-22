# xgboost (word11) — final

seed: 46 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}` | frozen hash: `da0a691809224d21cbd78e4be5a9b8f005d8ab0577a434b86344451b3e731d25` | val macro-F1: 0.7698

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7702 | 0.7698 | 0.7701 | 0.6560 | 0.0727 |
| test | 0.7803 | 0.7802 | 0.7803 | 0.6707 | 0.0779 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7287    0.8370    0.7791      1258
     Neutral     0.7966    0.7224    0.7577      1095
    Positive     0.8374    0.7731    0.8039      1106

    accuracy                         0.7803      3459
   macro avg     0.7876    0.7775    0.7802      3459
weighted avg     0.7850    0.7803    0.7803      3459
```
