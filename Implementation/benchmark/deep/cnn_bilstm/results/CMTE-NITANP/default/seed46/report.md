# cnn_bilstm (word)

config: `default` | seed: 46 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 5

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7881 | 0.7877 | 0.7879 | 0.6814 | 0.0661 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7668    0.8156    0.7904      1258
     Neutral     0.7775    0.7370    0.7567      1095
    Positive     0.8246    0.8074    0.8159      1106

    accuracy                         0.7881      3459
   macro avg     0.7896    0.7867    0.7877      3459
weighted avg     0.7886    0.7881    0.7879      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7920 | 0.7410 | True |
| 2 | 0.5808 | 0.7639 | True |
| 3 | 0.4790 | 0.7798 | True |
| 4 | 0.3976 | 0.7889 | True |
| 5 | 0.3262 | 0.7910 | True |
| 6 | 0.2580 | 0.7703 | False |
| 7 | 0.2021 | 0.7798 | False |
