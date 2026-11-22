# lightgbm (word11) — final

seed: 44 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 300}` | frozen hash: `d9b7bec53b7743e012bea51ee2186bfcb1a2841833819cabdffacbc538c6bd02` | val macro-F1: 0.7952

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7950 | 0.7952 | 0.7951 | 0.6918 | 0.0201 |
| test | 0.8049 | 0.8048 | 0.8048 | 0.7066 | 0.0132 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7907    0.8140    0.8022      1258
     Neutral     0.7908    0.7662    0.7783      1095
    Positive     0.8350    0.8327    0.8339      1106

    accuracy                         0.8049      3459
   macro avg     0.8055    0.8043    0.8048      3459
weighted avg     0.8049    0.8049    0.8048      3459
```
