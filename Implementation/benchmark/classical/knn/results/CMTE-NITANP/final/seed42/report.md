# knn (word11) — final

seed: 42 | dataset: CMTE-NITANP | split: B | text: `text_clean` | features: 16,892

params: `{'n_neighbors': 15}` | frozen hash: `33b259222f2989e7b4c25a10a3c31e9b1dea10c6ca7618d597fba369fa3bf1ac` | val macro-F1: 0.7663

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| val | 0.7670 | 0.7663 | 0.7668 | 0.6537 | 0.0559 |
| test | 0.7759 | 0.7755 | 0.7748 | 0.6671 | 0.0665 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8272    0.7003    0.7585      1258
     Neutral     0.7447    0.7726    0.7584      1095
    Positive     0.7607    0.8653    0.8096      1106

    accuracy                         0.7759      3459
   macro avg     0.7776    0.7794    0.7755      3459
weighted avg     0.7798    0.7759    0.7748      3459
```
