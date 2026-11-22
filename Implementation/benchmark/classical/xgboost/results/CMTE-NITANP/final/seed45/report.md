# xgboost (word11) — final

seed: 45 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}` | frozen hash: `da0a691809224d21cbd78e4be5a9b8f005d8ab0577a434b86344451b3e731d25` | val macro-F1: 0.7700

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7702 | 0.7700 | 0.7702 | 0.6560 | 0.0704 |
| test | 0.7774 | 0.7773 | 0.7773 | 0.6661 | 0.0750 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7298    0.8267    0.7753      1258
     Neutral     0.7939    0.7178    0.7540      1095
    Positive     0.8266    0.7803    0.8028      1106

    accuracy                         0.7774      3459
   macro avg     0.7835    0.7749    0.7773      3459
weighted avg     0.7811    0.7774    0.7773      3459
```
