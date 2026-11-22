# bilstm (word)

config: `default` | seed: 46 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 5

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7968 | 0.7969 | 0.7968 | 0.6946 | 0.0558 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7976    0.7893    0.7934      1258
     Neutral     0.7714    0.7826    0.7770      1095
    Positive     0.8214    0.8192    0.8203      1106

    accuracy                         0.7968      3459
   macro avg     0.7968    0.7971    0.7969      3459
weighted avg     0.7969    0.7968    0.7968      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.8052 | 0.7509 | True |
| 2 | 0.5740 | 0.7781 | True |
| 3 | 0.4802 | 0.7926 | True |
| 4 | 0.4060 | 0.7979 | True |
| 5 | 0.3441 | 0.7990 | True |
| 6 | 0.2836 | 0.7985 | False |
| 7 | 0.2312 | 0.7967 | False |
