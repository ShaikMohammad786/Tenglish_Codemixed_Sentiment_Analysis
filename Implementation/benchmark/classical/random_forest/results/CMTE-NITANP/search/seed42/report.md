# random_forest (word11) — search

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'max_depth': None, 'min_samples_leaf': 1}` | frozen hash: `None` | val macro-F1: 0.7873

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7878 | 0.7873 | 0.7876 | 0.6811 | 0.0313 |
| test | 0.7956 | 0.7954 | 0.7954 | 0.6928 | 0.0418 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7710    0.8188    0.7941      1258
     Neutral     0.7992    0.7489    0.7732      1095
    Positive     0.8222    0.8156    0.8189      1106

    accuracy                         0.7956      3459
   macro avg     0.7975    0.7944    0.7954      3459
weighted avg     0.7963    0.7956    0.7954      3459
```

## Search history (val)

| params | val macro-F1 |
|---|---:|
| `{'max_depth': None, 'min_samples_leaf': 1}` | 0.7873 |
| `{'max_depth': None, 'min_samples_leaf': 3}` | 0.7757 |
| `{'max_depth': 64, 'min_samples_leaf': 1}` | 0.7635 |
| `{'max_depth': 64, 'min_samples_leaf': 3}` | 0.7544 |
