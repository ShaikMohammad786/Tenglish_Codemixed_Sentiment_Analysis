# xgboost (word11) — final

seed: 43 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}` | frozen hash: `da0a691809224d21cbd78e4be5a9b8f005d8ab0577a434b86344451b3e731d25` | val macro-F1: 0.7702

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7707 | 0.7702 | 0.7705 | 0.6570 | 0.0698 |
| test | 0.7736 | 0.7736 | 0.7736 | 0.6607 | 0.0748 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7218    0.8291    0.7717      1258
     Neutral     0.7872    0.7096    0.7464      1095
    Positive     0.8335    0.7740    0.8026      1106

    accuracy                         0.7736      3459
   macro avg     0.7808    0.7709    0.7736      3459
weighted avg     0.7782    0.7736    0.7736      3459
```
