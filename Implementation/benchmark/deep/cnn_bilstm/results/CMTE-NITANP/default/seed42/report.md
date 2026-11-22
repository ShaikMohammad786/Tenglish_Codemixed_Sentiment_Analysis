# cnn_bilstm (word)

config: `default` | seed: 42 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 4

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7826 | 0.7833 | 0.7833 | 0.6741 | 0.0447 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7829    0.7798    0.7814      1258
     Neutral     0.7270    0.7927    0.7584      1095
    Positive     0.8478    0.7758    0.8102      1106

    accuracy                         0.7826      3459
   macro avg     0.7859    0.7828    0.7833      3459
weighted avg     0.7860    0.7826    0.7833      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7877 | 0.7289 | True |
| 2 | 0.5778 | 0.7545 | True |
| 3 | 0.4779 | 0.7613 | True |
| 4 | 0.3929 | 0.7717 | True |
| 5 | 0.3238 | 0.7670 | False |
| 6 | 0.2538 | 0.7695 | False |
