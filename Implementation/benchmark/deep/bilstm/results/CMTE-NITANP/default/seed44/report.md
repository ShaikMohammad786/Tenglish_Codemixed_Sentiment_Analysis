# bilstm (word)

config: `default` | seed: 44 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 5

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.8028 | 0.8029 | 0.8029 | 0.7042 | 0.0547 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7576    0.8474    0.8000      1258
     Neutral     0.8180    0.7553    0.7854      1095
    Positive     0.8492    0.7993    0.8235      1106

    accuracy                         0.8028      3459
   macro avg     0.8083    0.8006    0.8029      3459
weighted avg     0.8060    0.8028    0.8029      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7977 | 0.7367 | True |
| 2 | 0.5692 | 0.7681 | True |
| 3 | 0.4789 | 0.7847 | True |
| 4 | 0.4067 | 0.7919 | True |
| 5 | 0.3451 | 0.7940 | True |
| 6 | 0.2851 | 0.7898 | False |
| 7 | 0.2335 | 0.7909 | False |
