# cnn_bilstm (word)

config: `default` | seed: 45 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 6

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7809 | 0.7812 | 0.7812 | 0.6719 | 0.0968 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7293    0.8394    0.7805      1258
     Neutral     0.7734    0.7388    0.7557      1095
    Positive     0.8663    0.7559    0.8073      1106

    accuracy                         0.7809      3459
   macro avg     0.7897    0.7780    0.7812      3459
weighted avg     0.7871    0.7809    0.7812      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7811 | 0.7278 | True |
| 2 | 0.5801 | 0.7581 | True |
| 3 | 0.4809 | 0.7710 | True |
| 4 | 0.3963 | 0.7722 | True |
| 5 | 0.3288 | 0.7749 | True |
| 6 | 0.2586 | 0.7770 | True |
| 7 | 0.1993 | 0.7748 | False |
| 8 | 0.1546 | 0.7667 | False |
