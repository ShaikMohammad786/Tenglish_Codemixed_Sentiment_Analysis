# cnn_bilstm (word)

config: `default` | seed: 44 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 5

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7817 | 0.7809 | 0.7811 | 0.6728 | 0.0746 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7398    0.8339    0.7840      1258
     Neutral     0.8059    0.7014    0.7500      1095
    Positive     0.8153    0.8020    0.8086      1106

    accuracy                         0.7817      3459
   macro avg     0.7870    0.7791    0.7809      3459
weighted avg     0.7848    0.7817    0.7811      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7776 | 0.7200 | True |
| 2 | 0.5768 | 0.7596 | True |
| 3 | 0.4788 | 0.7567 | False |
| 4 | 0.3988 | 0.7703 | True |
| 5 | 0.3275 | 0.7853 | True |
| 6 | 0.2594 | 0.7658 | False |
| 7 | 0.2049 | 0.7770 | False |
