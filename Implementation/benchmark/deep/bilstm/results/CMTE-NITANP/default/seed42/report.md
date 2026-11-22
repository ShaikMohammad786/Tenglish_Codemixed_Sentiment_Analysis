# bilstm (word)

config: `default` | seed: 42 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 6

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8025 | 0.8026 | 0.8025 | 0.7031 | 0.0670 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7798    0.8164    0.7977      1258
     Neutral     0.8042    0.7653    0.7843      1095
    Positive     0.8282    0.8237    0.8259      1106

    accuracy                         0.8025      3459
   macro avg     0.8041    0.8018    0.8026      3459
weighted avg     0.8030    0.8025    0.8025      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7969 | 0.7367 | True |
| 2 | 0.5734 | 0.7719 | True |
| 3 | 0.4839 | 0.7838 | True |
| 4 | 0.4096 | 0.7903 | True |
| 5 | 0.3474 | 0.7981 | True |
| 6 | 0.2921 | 0.7996 | True |
| 7 | 0.2395 | 0.7823 | False |
| 8 | 0.1977 | 0.7893 | False |
