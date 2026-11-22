# cnn_bilstm (word)

config: `default` | seed: 43 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 4

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7895 | 0.7900 | 0.7900 | 0.6839 | 0.0383 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7670    0.8140    0.7898      1258
     Neutral     0.7549    0.7735    0.7641      1095
    Positive     0.8583    0.7776    0.8159      1106

    accuracy                         0.7895      3459
   macro avg     0.7934    0.7884    0.7900      3459
weighted avg     0.7924    0.7895    0.7900      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7930 | 0.7254 | True |
| 2 | 0.5780 | 0.7590 | True |
| 3 | 0.4750 | 0.7696 | True |
| 4 | 0.3982 | 0.7826 | True |
| 5 | 0.3247 | 0.7754 | False |
| 6 | 0.2596 | 0.7760 | False |
